"""Restore pre-classifier full-image labels using historical per-image lineage."""
import argparse, csv, hashlib, json, sys
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / 'Phase_detection'
sys.path.insert(0, str(PHASE))
from stage1_python_worker.obb.geometry import aligned_crop, rectangle_from_mask
from stage1_python_worker.obb.inference import detections_from_result
from stage1_python_worker.obb.integrated import rectify_full_image, rotate_180


def read_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    home = ROOT / 'Phase_orientation_classifier'
    audit = home / 'outputs/provenance_audit'
    manifest = json.loads((home/'dataset_2000/manifest.json').read_text())
    lineage = {r['id']: r for r in json.loads((audit/'crop_lineage.json').read_text())}
    oldbase = PHASE/'outputs/orientation_evaluation/dryad_2000_seed42'
    old = json.loads((oldbase/'manifest_1000.json').read_text())['samples']
    old_by_id = {Path(r['source_relative_path']).stem: r for r in old}
    p2 = json.loads((ROOT/'Phase_single_digit_BB/outputs/exp0_pretrained_obb/selected_samples.json').read_text())['samples']
    p2_by_id = {Path(r['original_image_path']).stem: r for r in p2}
    rfbase = PHASE/'outputs/pilots/roboflow_failures13_20261027'
    rf = {r['example_id'].removeprefix('train_'):r for r in csv.DictReader((rfbase/'orientation_13_predictions.csv').open())}
    rfgeo = {r['example_id'].removeprefix('train_'):r for r in json.loads((rfbase/'selection_and_geometry.json').read_text())['samples']}
    dryad = Path('/Users/ququ/Downloads/doi_10_5061_dryad_7d7wm3860__v20260130/Word-Wheel_Water_Meter_Dataset/detection/train')
    weights = PHASE/'outputs/stage1_obb/final_run/artifacts/runs/register_yolo11n_obb/weights/best.pt'
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == 'e7c03f5e8bd7e09b00fdd0d82f4d232f641ea576566f38b672503b5ad503e07e'
    model = None
    torch.set_num_threads(4); cv2.setNumThreads(2)
    for folder in ['images', 'records']:
        (args.output/folder).mkdir(parents=True, exist_ok=True)
    done = 0
    for number, row in enumerate(manifest['records'],1):
        rid = row['original_id']; recordpath=args.output/'records'/f'{rid}.json'
        if recordpath.exists(): continue
        item={'id':rid,'index':number,'excluded_by_previous_review':row['user_excluded'],
              'reviewed_crop_label': None if row['user_excluded'] else int(row['orientation_label']=='upside_down'),
              'label_stage':'geometrically_rectified_full_image_before_legacy_orientation_classifier',
              'original_id_mapping':row['source_mapping_method'],'reviewed_crop_path':row['packaged_image_path']}
        try:
            branch=lineage.get(rid,{}).get('lineage');meta=row.get('stage1_processing')
            q=None;historical_flip=False;pre=None;validation=None
            if meta:
                branch=meta['crop_source']
                rec=meta if branch!='prior_stage1_crop' else json.loads((PHASE/'outputs/clear_blurry_review_seed42/records'/f'{rid}.json').read_text())
                label=rec['orientation'].strip().lower(); assert label in ('correct','upside_down')
                historical_flip=label=='upside_down'; rgb=read_rgb(dryad/'train_img'/f'{rid}.png')
                if rec.get('fallback')=='manual_segmentation_mask':
                    with Image.open(dryad/'train_seg_label'/f'{rid}.png') as im:q=rectangle_from_mask(np.asarray(im.convert('L')))
                else:
                    if model is None: model=YOLO(str(weights),task='obb')
                    conf=.05 if branch=='stage1_detector_low_confidence' or rec.get('fallback')=='confidence_threshold_0.05' else .25
                    pred=model.predict(rgb[:,:,::-1].copy(),imgsz=640,conf=conf,iou=.7,device='cpu',verbose=False)[0]
                    boxes=sorted(detections_from_result(pred),key=lambda b:b['confidence'],reverse=True)
                    if not boxes: raise ValueError('Historical detector branch returned no box')
                    q=np.asarray(boxes[0]['corners_xy'])
                pre,q1,matrix,angle=rectify_full_image(rgb,q)
                final,q2,_=rotate_180(pre,q1) if historical_flip else (pre,q1,None)
                replay,_=aligned_crop(final,q2,.03)
                reference=read_rgb(home/'dataset_2000'/row['packaged_image_path'])
                if reference.shape==replay.shape:
                    err=np.abs(reference.astype(float)-replay.astype(float))
                    validation={'same_shape':True,'max_pixel_error':float(err.max()),'mean_pixel_error':float(err.mean())}
                else: validation={'same_shape':False,'old_shape':list(reference.shape),'new_shape':list(replay.shape)}
                item.update(rotation_angle_degrees=angle,affine_matrix=matrix.tolist(),corners_xy=q.tolist())
            elif branch in ('qualified_exact_phase2','qualified_exact_corrected_full','qualified_near_identical_corrected_full'):
                rec=old_by_id[rid]
                assert rec['applied_transform'] in ('identity','rotate_180')
                historical_flip=rec['applied_transform']=='rotate_180'
                reference_path=oldbase/rec['reference_path']
                if reference_path.is_file():pre=read_rgb(reference_path)
                else:
                    corrected=read_rgb(oldbase/rec['prediction_adjusted_path']);pre=cv2.rotate(corrected,cv2.ROTATE_180) if historical_flip else corrected
                if branch=='qualified_exact_phase2':
                    original=p2_by_id[rid];assert original['phase1_orientation_prediction'].lower()==rec['normalized_prediction'].lower()
                item.update(saved_full_image_sample_id=rec['sample_id'],saved_calibration_affine=rec['calibration_affine'])
            elif branch=='qualified_exact_integration':
                folder=Path(lineage[rid]['source_artifacts'][0]['path']).parent
                rec=json.loads((folder/'result.json').read_text());historical_flip=rec['rotation_180_applied'];pre=read_rgb(folder/'rectified_full.png')
            elif branch=='qualified_exact_roboflow13':
                rec=rf[rid];historical_flip=rec['rotation_180_applied'].lower()=='true';rgb=read_rgb(dryad/'train_img'/f'{rid}.png')
                pre,_,matrix,angle=rectify_full_image(rgb,np.asarray(rfgeo[rid]['corners_xy']));item.update(rotation_angle_degrees=angle,affine_matrix=matrix.tolist())
            elif branch=='qualified_gt_replay' or branch in ('qualified_exact_train_gt_original','qualified_exact_val_gt_original'):
                if branch=='qualified_gt_replay':
                    info=lineage[rid]['saved_geometry'];path=Path(info['path']);historical_flip=info['rotation']==180
                else:
                    split='train' if branch=='qualified_exact_train_gt_original' else 'val'
                    path=PHASE/'dataset/stageOneOBB/images'/split/f'train_{rid}.jpg'
                rgb=read_rgb(path);vals=(path.parents[2]/'labels'/path.parent.name/(path.stem+'.txt')).read_text().split();assert len(vals)==9
                q=np.asarray(list(map(float,vals[1:]))).reshape(4,2)*[rgb.shape[1],rgb.shape[0]]
                pre,_,matrix,angle=rectify_full_image(rgb,q);item.update(rotation_angle_degrees=angle,affine_matrix=matrix.tolist(),historical_geometry_image=str(path))
            else: raise ValueError(f'Unresolved historical branch {branch}')
            label=None if row['user_excluded'] else item['reviewed_crop_label'] ^ int(historical_flip)
            target=args.output/'images'/f'{rid}.png'
            Image.fromarray(pre).save(target)
            item.update(status='excluded' if row['user_excluded'] else 'restored',restored_label=label,
                        historical_extra_180=bool(historical_flip),lineage=branch,image_path=str(target.relative_to(args.output)),
                        image_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),image_size_wh=[pre.shape[1],pre.shape[0]],replay_validation=validation)
        except Exception as exc:
            item.update(status='failed',restored_label=None,error=f'{type(exc).__name__}: {exc}')
        tmp=recordpath.with_suffix('.tmp');tmp.write_text(json.dumps(item,indent=2)+'\n');tmp.replace(recordpath)
        done+=1
        if done%50==0: print(f'Processed {done}; latest {rid}: {item["status"]}',flush=True)
        if args.limit and done>=args.limit:break
    print(f'Completed {done} new records',flush=True)
if __name__=='__main__':main()
