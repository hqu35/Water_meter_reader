"""Real-data register localization. Each phase is explicit; train requires a review gate.

python scripts/stage1.py --root /content/cs539_stage1 inspect|prepare|approve|train|evaluate|package
See docs/STAGE1.md. Dataset code is treated as data and is never executed.
"""
from pathlib import Path, PurePosixPath
from collections import Counter
import argparse, hashlib, io, json, math, os, random, shutil, subprocess, sys, time, zipfile

REPO=Path(__file__).resolve().parents[1]
CONFIG=json.loads((REPO/'configs/stage1.json').read_text())

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def dump(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')

def archive(root):return root/'raw/Word-Wheel_Water_Meter_Dataset.zip'
def artifacts(root):return root/'artifacts'

def gpu(root):
 import torch
 assert torch.cuda.is_available(),'CUDA unavailable; CPU fallback is intentionally disabled'
 name=torch.cuda.get_device_name(0);assert 'T4' in name,name
 x=torch.arange(1024,dtype=torch.float32).reshape(32,32)
 y=x.cuda()@x.cuda().T;torch.cuda.synchronize()
 assert torch.allclose(y.cpu(),x@x.T,rtol=1e-5,atol=1e-3)
 report=dict(gpu=name,torch=torch.__version__,cuda=torch.version.cuda,python=sys.version,cuda_matmul_passed=True,
             nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
 dump(artifacts(root)/'gpu.json',report);print(json.dumps(report,indent=2))

def download(root):
 import requests
 p=archive(root);p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists() or p.stat().st_size!=CONFIG['dataset_bytes']:
  tmp=p.with_suffix('.partial')
  with requests.get('https://datadryad.org/downloads/file_stream/'+str(CONFIG['dataset_file_id']),stream=True,timeout=(30,180)) as r:
   r.raise_for_status()
   with tmp.open('wb') as f:
    for chunk in r.iter_content(8*1024*1024):f.write(chunk)
  tmp.replace(p)
 assert sha(p)==CONFIG['dataset_sha256'],'Archive checksum mismatch'
 dump(artifacts(root)/'download.json',dict(doi=CONFIG['dataset_doi'],version=CONFIG['dataset_version_id'],sha256=sha(p),bytes=p.stat().st_size,checksum_verified=True))

def discover(z):
 """Pair by exact archive parent + filename, never positional glob ordering."""
 names=set(z.namelist());out={'train':[],'test':[]};orphans=[]
 for name in sorted(names):
  p=PurePosixPath(name)
  image_dir=p.parent.name.lower()
  if p.suffix.lower() not in ['.png','.jpg','.jpeg'] or image_dir not in ['det_img','train_img','test_img']:continue
  split='train' if any('train' in x.lower() for x in p.parts[:-2]) else 'test' if any('test' in x.lower() for x in p.parts[:-2]) else None
  if split is None:raise ValueError('Cannot determine official split: '+name)
  mask_dir={'det_img':'seg_label','train_img':'train_seg_label','test_img':'test_seg_label'}[image_dir]
  mask=str(p.parent.parent/mask_dir/p.name)
  if mask not in names:orphans.append({'image':name,'expected_mask':mask});continue
  out[split].append((name,mask))
 if orphans:raise ValueError('Unpaired images: '+str(orphans[:10]))
 assert out['train'] and out['test'],'Detection image directories not found'
 return out

def decode(z,imname,maskname):
 import numpy as np
 from PIL import Image
 imraw=z.read(imname);maskraw=z.read(maskname)
 source=Image.open(io.BytesIO(imraw));image_format=source.format
 image=source.convert('RGB');maskim=Image.open(io.BytesIO(maskraw));mask=np.asarray(maskim)
 assert mask.shape[:2]==(image.height,image.width),('Image/mask dimension mismatch',imname,mask.shape,image.size)
 if mask.ndim==3:
  assert np.array_equal(mask[...,0],mask[...,1]) and np.array_equal(mask[...,0],mask[...,2]),('Unexpected color mask',maskname)
  mask=mask[...,0]
 vals=np.unique(mask).tolist()
 assert set(vals)<=set([0,1,255]),('Unexpected mask encoding',maskname,vals)
 fg=mask>0
 pixelhash=hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
 return image,fg,dict(image_sha256=hashlib.sha256(imraw).hexdigest(),image_format=image_format,mask_sha256=hashlib.sha256(maskraw).hexdigest(),pixel_sha256=pixelhash,mask_values=vals)

def bbox_from_mask(fg):
 import numpy as np
 yy,xx=np.nonzero(fg)
 if len(xx)==0:return None
 # Half-open pixel bounds preserve a one-pixel object and the far image edge.
 return [int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1]

def normalize_box(box,w,h):
 x0,y0,x1,y1=box
 assert 0<=x0<x1<=w and 0<=y0<y1<=h
 return [(x0+x1)/(2*w),(y0+y1)/(2*h),(x1-x0)/w,(y1-y0)/h]

def inspect(root):
 from PIL import Image
 import numpy as np
 assert sha(archive(root))==CONFIG['dataset_sha256'],'Archive checksum mismatch'
 with zipfile.ZipFile(archive(root)) as z:
  pairs=discover(z);sample=[]
  for split,ps in pairs.items():
   for i,m in ps[:8]:
    image,fg,h=decode(z,i,m);sample.append(dict(split=split,image=i,mask=m,size=image.size,image_format=h['image_format'],mask_values=h['mask_values'],foreground_pixels=int(fg.sum()),bbox=bbox_from_mask(fg)))
  report=dict(archive_sha256=CONFIG['dataset_sha256'],archive_members=len(z.namelist()),official_counts={k:len(v) for k,v in pairs.items()},pairing='exact split/parent and filename',sample=sample,top_level=sorted({PurePosixPath(n).parts[0] for n in z.namelist() if PurePosixPath(n).parts}))
 dump(artifacts(root)/'inspection.json',report);print(json.dumps(report,indent=2))

def prepare(root):
 import numpy as np
 from PIL import Image,ImageDraw,ImageFont
 import yaml
 assert sha(archive(root))==CONFIG['dataset_sha256'],'Archive checksum mismatch'
 target=root/'prepared';art=artifacts(root);rng=random.Random(CONFIG['seed']);rows=[];seen={};excluded=[]
 if (target/'images').exists():raise RuntimeError('Use a fresh prepared directory; refuse to mix stale images into a new subset.')
 # Official test is selected first; train/val never borrow from it.
 with zipfile.ZipFile(archive(root)) as z:
  pools=discover(z)
  for src,count,assign in [('test',CONFIG['test_count'],lambda k:'test'),('train',CONFIG['train_count']+CONFIG['val_count'],lambda k:'val' if k<CONFIG['val_count'] else 'train')]:
   candidates=list(pools[src]);rng.shuffle(candidates);accepted=0
   for imname,maskname in candidates:
    image,fg,hashes=decode(z,imname,maskname);box=bbox_from_mask(fg)
    if box is None:excluded.append(dict(image=imname,reason='empty_foreground_mask'));continue
    pixelhash=hashes['pixel_sha256']
    if pixelhash in seen:
     excluded.append(dict(image=imname,reason='duplicate_decoded_image',duplicate_of=seen[pixelhash]));continue
    seen[pixelhash]=imname;split=assign(accepted);stem=Path(imname).stem;uid=f'{src}_{stem}'
    ext='.jpg' if hashes['image_format']=='JPEG' else '.png'
    image_file=uid+ext
    imagepath=target/'images'/split/image_file;labelpath=target/'labels'/split/(uid+'.txt')
    imagepath.parent.mkdir(parents=True,exist_ok=True);labelpath.parent.mkdir(parents=True,exist_ok=True)
    # Source image files are named .png but actually JPEG in the frozen archive.
    # Preserve original encoded bytes without recompression; normalize extension.
    imagepath.write_bytes(z.read(imname))
    norm=normalize_box(box,image.width,image.height);label='0 '+' '.join(f'{v:.10f}' for v in norm)+'\n';labelpath.write_text(label)
    # Round-trip YOLO coordinates must recover exact half-open bounds.
    cx,cy,bw,bh=map(float,label.split()[1:]);decoded=np.array([(cx-bw/2)*image.width,(cy-bh/2)*image.height,(cx+bw/2)*image.width,(cy+bh/2)*image.height])
    assert float(np.abs(decoded-np.array(box)).max())<1e-5
    x0,y0,x1,y1=box;assert int(fg[y0:y1,x0:x1].sum())==int(fg.sum())
    rows.append(dict(id=uid,image_file=image_file,split=split,official_split=src,source_image=imname,source_mask=maskname,width=image.width,height=image.height,box_xyxy=box,yolo=norm,foreground_pixels=int(fg.sum()),foreground_area_ratio=float(fg.mean()),label_sha256=sha(labelpath),**hashes))
    accepted+=1
    if accepted==count:break
   assert accepted==count,(src,accepted,count)
 manifest=art/'split_manifest.json';dump(manifest,dict(seed=CONFIG['seed'],dataset_sha256=CONFIG['dataset_sha256'],records=rows,exclusions=excluded))
 (target/'data.yaml').write_text(yaml.safe_dump(dict(path=str(target.resolve()),train='images/train',val='images/val',test='images/test',names={0:'register'}),sort_keys=False))
 # 32 review images spanning all splits and mask area extremes, plus fixed random samples.
 selected=[]
 for split,n in [('train',16),('val',8),('test',8)]:
  rr=[r for r in rows if r['split']==split];byarea=sorted(rr,key=lambda x:x['foreground_area_ratio'])
  ids={byarea[0]['id'],byarea[-1]['id']};others=[r for r in rr if r['id'] not in ids]
  chosen=[byarea[0],byarea[-1]]+rng.sample(others,n-2);selected+=chosen
 sheets=[];vis=art/'annotations';vis.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(archive(root)) as z:
  for sheet_i in range(4):
   sheet=Image.new('RGB',(1536,1680),'#f5f5f5');dr=ImageDraw.Draw(sheet)
   group=selected[sheet_i*8:(sheet_i+1)*8]
   for j,r in enumerate(group):
    image,fg,_=decode(z,r['source_image'],r['source_mask']);over=np.array(image).copy();over[fg]=(over[fg]*.55+np.array([50,240,90])*.45).astype('uint8')
    overlay=Image.fromarray(over);od=ImageDraw.Draw(overlay);od.rectangle((r['box_xyxy'][0],r['box_xyxy'][1],r['box_xyxy'][2]-1,r['box_xyxy'][3]-1),outline=(255,40,40),width=3)
    x=(j%2)*768;y=(j//2)*420
    sheet.paste(image.resize((384,384)),(x,y+32));sheet.paste(overlay.resize((384,384)),(x+384,y+32))
    dr.text((x+7,y+9),r['split']+' / '+r['id']+'     ORIGINAL | MASK + YOLO BOX',fill='#111111')
   p=vis/f'review_{sheet_i+1:02d}.jpg';sheet.save(p,quality=94);sheets.append(dict(path=str(p.relative_to(art)),sha256=sha(p),ids=[r['id'] for r in group]))
 dump(art/'validation.json',dict(passed=True,conversion='union of all nonzero mask pixels; max+1 exclusive bounds; class 0 register',mask_values=sorted({v for r in rows for v in r['mask_values']}),counts=dict(Counter(r['split'] for r in rows)),manifest_sha256=sha(manifest),exact_decoded_image_deduplication=True,all_selected_dimensions_aligned=True,all_boxes_cover_foreground=True,yolo_roundtrip_tolerance_pixels=1e-5,review_image_count=len(selected),sheets=sheets,excluded_count=len(excluded)))
 print('PREPARATION_COMPLETE: 32 review images saved; training is still locked pending visual review.')

def approve(root,reviewer):
 art=artifacts(root);report=json.loads((art/'validation.json').read_text());assert report['passed'];assert report['review_image_count']>=30
 assert sha(art/'split_manifest.json')==report['manifest_sha256']
 for s in report['sheets']:assert sha(art/s['path'])==s['sha256']
 dump(art/'annotation_review.json',dict(approved=True,reviewer=reviewer,reviewed_image_count=report['review_image_count'],manifest_sha256=report['manifest_sha256'],reviewed_sheets=report['sheets'],time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))

def gate(root):
 art=artifacts(root);review=json.loads((art/'annotation_review.json').read_text());manifest=art/'split_manifest.json'
 assert review['approved'] and review['reviewed_image_count']>=30
 assert sha(manifest)==review['manifest_sha256'],'Manifest changed after review'
 for s in review['reviewed_sheets']:assert sha(art/s['path'])==s['sha256']
 data=json.loads(manifest.read_text());hashes=[]
 for r in data['records']:
  image=root/'prepared/images'/r['split']/r['image_file'];label=root/'prepared/labels'/r['split']/(r['id']+'.txt')
  assert sha(label)==r['label_sha256'],'Label changed after review'
  from PIL import Image
  import numpy as np
  assert hashlib.sha256(np.array(Image.open(image).convert('RGB')).tobytes()).hexdigest()==r['pixel_sha256'],'Image changed after review'
  hashes.append(r['pixel_sha256'])
 assert len(hashes)==len(set(hashes)),'Image leakage across splits'
 return review

def train(root):
 os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['YOLO_CONFIG_DIR']=str(root/'ultralytics-settings')
 review=gate(root);gpu(root)
 import torch,ultralytics
 from ultralytics import YOLO
 assert ultralytics.__version__=='8.4.0',ultralytics.__version__
 art=artifacts(root);dump(art/'training_config.json',CONFIG)
 (art/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
 model=YOLO(CONFIG['model']);model.add_callback('on_train_epoch_end',lambda trainer:print('STAGE1_EPOCH',trainer.epoch+1,flush=True))
 dump(art/'source_checksums.json',{str(p.relative_to(REPO)):sha(p) for p in [REPO/'scripts/stage1.py',REPO/'configs/stage1.json',REPO/'requirements-stage1.txt']})
 dump(art/'pretrained.json',dict(model=CONFIG['model'],sha256=sha(model.ckpt_path)))
 result=model.train(data=str(root/'prepared/data.yaml'),epochs=CONFIG['epochs'],imgsz=CONFIG['imgsz'],batch=CONFIG['batch'],device=0,workers=CONFIG['workers'],seed=CONFIG['seed'],deterministic=True,
   optimizer=CONFIG['optimizer'],lr0=CONFIG['lr0'],warmup_bias_lr=CONFIG['warmup_bias_lr'],warmup_epochs=CONFIG['warmup_epochs'],patience=CONFIG['patience'],project=str(art/'runs'),name='register_yolo11n',exist_ok=False,
   pretrained=True,amp=False,cache=False,plots=True,save=True,save_period=5,single_cls=True,
   mosaic=0.0,mixup=0.0,cutmix=0.0,copy_paste=0.0,erasing=0.0,fliplr=0.0,flipud=0.0,
   degrees=0.0,translate=0.0,scale=0.0,shear=0.0,perspective=0.0,
   hsv_h=0.0,hsv_s=0.0,hsv_v=0.0,close_mosaic=0)
 dump(art/'training_complete.json',dict(save_dir=str(model.trainer.save_dir),best=str(model.trainer.best),last=str(model.trainer.last),metrics=result.results_dict))
 normalize_checkpoints(root)
 print('TRAINING_COMPLETE',flush=True)

def normalize_checkpoints(root):
 """single_cls training can name class zero 'item'; normalize metadata only.

 Round-trip and compare every tensor before atomically replacing a checkpoint.
 Optimizer state, epoch, architecture and all learned weights are retained.
 """
 import torch
 art=artifacts(root);done=json.loads((art/'training_complete.json').read_text());reports=[]
 for p in sorted(Path(done['save_dir']).joinpath('weights').glob('*.pt')):
  ckpt=torch.load(p,map_location='cpu',weights_only=False);states={}
  for slot in ['model','ema']:
   obj=ckpt.get(slot)
   if obj is None:continue
   assert len(obj.names)==1,'Unexpected class count'
   states[slot]={k:v.detach().clone() for k,v in obj.state_dict().items()}
   obj.names={0:'register'}
  before=sha(p);tmp=p.with_suffix('.metadata.tmp');torch.save(ckpt,tmp)
  restored=torch.load(tmp,map_location='cpu',weights_only=False)
  for slot,state in states.items():
   assert restored[slot].names=={0:'register'}
   for key,value in state.items():assert torch.equal(value,restored[slot].state_dict()[key]),('Tensor changed',p,key)
  tmp.replace(p);reports.append(dict(file=str(p),before_sha256=before,after_sha256=sha(p),all_model_tensors_identical=True,names={0:'register'}))
 dump(art/'checkpoint_class_names.json',dict(reason='Normalize framework single_cls display name item to register; no retraining or tensor changes',source_sha256=sha(__file__),checkpoints=reports))
 print('CHECKPOINT_CLASS_NAMES_VERIFIED',len(reports),flush=True)

def prediction_batches(model,rows,image_root,batch_size=32):
 """Ultralytics treats an in-memory image list as one batch; bound each call.

 Preserve the manifest order and verify filenames so boxes cannot be evaluated
 against another image's ground truth after a loader reordering.
 """
 for start in range(0,len(rows),batch_size):
  group=rows[start:start+batch_size]
  predictions=list(model.predict(source=[str(image_root/r['image_file']) for r in group],device=0,imgsz=CONFIG['imgsz'],batch=batch_size,conf=.25,iou=.7,stream=True,verbose=False))
  assert len(predictions)==len(group),'Prediction count mismatch'
  for row,pred in zip(group,predictions):
   assert Path(pred.path).name==row['image_file'],'Prediction/ground-truth image ordering mismatch'
   yield row,pred

def evaluate(root):
 from ultralytics import YOLO
 import numpy as np
 from PIL import Image,ImageDraw
 art=artifacts(root);done=json.loads((art/'training_complete.json').read_text());model=YOLO(done['best'])
 assert model.names=={0:'register'},'Run normalize-checkpoints before evaluation'
 result=model.val(data=str(root/'prepared/data.yaml'),split='test',device=0,imgsz=CONFIG['imgsz'],batch=CONFIG['batch'],workers=CONFIG['workers'],plots=True,save_json=True,project=str(art/'evaluation'),name='heldout_test',exist_ok=False,conf=.001,iou=.7)
 dump(art/'heldout_metrics.json',dict(split='test',source='official Dryad detection test pool only',count=CONFIG['test_count'],selection='never used for fitting, early stopping or threshold selection',metrics=result.results_dict,speed=result.speed,checkpoint_sha256=sha(done['best']),evaluation_source_sha256=sha(__file__)))
 # Fixed operating point, separate from the low-confidence AP calculation.
 rows=[r for r in json.loads((art/'split_manifest.json').read_text())['records'] if r['split']=='test'];threshold=.25;records=[]
 for row,pred in prediction_batches(model,rows,root/'prepared/images/test'):
  boxes=pred.boxes.xyxy.cpu().numpy();scores=pred.boxes.conf.cpu().numpy();gt=np.array(row['box_xyxy']);ious=[]
  for b in boxes:
   inter=np.maximum(0,np.minimum(b[2:],gt[2:])-np.maximum(b[:2],gt[:2])).prod();union=np.maximum(0,b[2:]-b[:2]).prod()+(gt[2:]-gt[:2]).prod()-inter;ious.append(float(inter/union))
  best=max(ious,default=0);records.append(dict(id=row['id'],image_file=row['image_file'],gt_box=row['box_xyxy'],boxes=boxes.tolist(),confidences=scores.tolist(),best_iou=best,matched=best>=.5))
 assert len(records)==len(rows)==CONFIG['test_count'],'Incomplete held-out predictions'
 dump(art/'test_predictions.json',dict(confidence_threshold=threshold,iou_threshold=.5,records=records))
 # Include failures rather than only attractive predictions.
 selected=sorted(records,key=lambda r:r['best_iou'])[:8]+random.Random(CONFIG['seed']).sample(records,8)
 for si in range(2):
  sheet=Image.new('RGB',(1600,1800),'white');d=ImageDraw.Draw(sheet)
  for j,r in enumerate(selected[si*8:(si+1)*8]):
   im=Image.open(root/'prepared/images/test'/r['image_file']).convert('RGB');dr=ImageDraw.Draw(im);dr.rectangle(r['gt_box'],outline='#28e43e',width=3)
   for b in r['boxes']:dr.rectangle(b,outline='#ff4030',width=3)
   x=(j%2)*800;y=(j//2)*450;sheet.paste(im.resize((416,416)),(x+180,y+28));d.text((x+15,y+7),r['id']+f" / IoU {r['best_iou']:.3f} / GREEN GT; RED PRED",fill='black')
  sheet.save(art/f'test_predictions_{si+1:02d}.jpg',quality=93)
 dump(art/'operating_point.json',dict(confidence=threshold,iou=.5,images=len(records),localized=sum(r['matched'] for r in records),image_localization_recall=sum(r['matched'] for r in records)/len(records),missed=sum(not r['matched'] for r in records),total_predictions=sum(len(r['boxes']) for r in records)))
 print('EVALUATION_COMPLETE',json.dumps(result.results_dict),flush=True)

def package(root):
 art=artifacts(root);assert (art/'heldout_metrics.json').exists()
 files={str(p.relative_to(art)):sha(p) for p in art.rglob('*') if p.is_file() and p.name!='artifact_checksums.json'};dump(art/'artifact_checksums.json',files)
 p=root/'stage1_results.zip'
 with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
  for f in art.rglob('*'):
   if f.is_file():z.write(f,str(f.relative_to(root)))
 dump(root/'results_bundle.json',dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
 print((root/'results_bundle.json').read_text())

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=REPO/'outputs/stage1');p.add_argument('phase',choices=['gpu','download','inspect','prepare','approve','train','normalize-checkpoints','evaluate','package']);p.add_argument('--reviewer');a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
 if a.phase=='approve':
  assert a.reviewer,'Explicit reviewer and completed visual inspection required';approve(a.root,a.reviewer)
 else:globals()[a.phase.replace('-','_')](a.root)

if __name__=='__main__':main()
