"""Evaluate the finalized validation-selected checkpoint on frozen test images."""
import argparse
import collections
import csv
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from .data import load_dataset, verify_manifest
from .model import OrientationPredictor, build, letterbox


def describe(matrix):
    return {
        'orientation_samples': int(matrix.sum()),
        'accuracy': float(np.trace(matrix)/matrix.sum()),
        'confusion_matrix': matrix.tolist(),
        'per_class': {
            str(i): {'name': 'upright' if i==0 else 'upside_down',
                     'precision': float(matrix[i,i]/max(1,matrix[:,i].sum())),
                     'recall': float(matrix[i,i]/max(1,matrix[i,:].sum())),
                     'support': int(matrix[i,:].sum())} for i in range(2)}
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('--batch-size',type=int,default=16)
    args=parser.parse_args();run=args.run
    if (run/'test_metrics.json').exists():raise FileExistsError('Test metrics already exist; refusing repeated test evaluation')
    if not torch.backends.mps.is_available():raise RuntimeError('MPS is required')
    torch.set_num_threads(4)
    checkpoint=torch.load(run/'best.pt',map_location='mps',weights_only=False)
    manifest=json.loads((run/'split_manifest.json').read_text());verify_manifest(manifest)
    with (run/'metrics.csv').open() as f:history=list(csv.DictReader(f))
    best=max(history,key=lambda r:float(r['val_accuracy']))
    if checkpoint['epoch']!=int(best['epoch']):raise ValueError('Checkpoint differs from the validation-selected epoch')
    model=build(False).to('mps');model.load_state_dict(checkpoint['model']);model.eval()
    dataset=load_dataset(manifest,'test',model)
    if not len(dataset):raise ValueError('Empty test split')
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=0)
    matrix=np.zeros((2,2),dtype=int);by_source=collections.defaultdict(lambda:np.zeros((2,2),dtype=int))
    predictions=[];cursor=0
    with torch.inference_mode():
        for batch,(x,y) in enumerate(loader,1):
            probs=model(x.to('mps')).softmax(1).cpu();pred=probs.argmax(1)
            for offset,(actual,guess) in enumerate(zip(y.tolist(),pred.tolist())):
                index=cursor+offset;row=dataset.records[index//2]
                matrix[actual,guess]+=1;by_source[row['source']][actual,guess]+=1
                predictions.append({'path':row['path'],'source':row['source'],'group_id':row['group_id'],
                  'virtual_180':bool(index%2),'actual':actual,'predicted':guess,
                  'confidence':float(probs[offset,guess]),'correct':actual==guess})
            cursor+=len(y)
            if batch==1 or batch%20==0 or batch==len(loader):print(f'Test batch {batch}/{len(loader)}',flush=True)
        x=dataset[0][0].unsqueeze(0).to('mps')
        for _ in range(10):model(x)
        torch.mps.synchronize();latencies=[]
        for _ in range(100):
            start=time.perf_counter();model(x);torch.mps.synchronize();latencies.append((time.perf_counter()-start)*1000)
    summary=describe(matrix)
    summary.update(checkpoint='best.pt',checkpoint_sha256=hashlib.sha256((run/'best.pt').read_bytes()).hexdigest(),
      selected_epoch=checkpoint['epoch'],validation_accuracy=checkpoint['val_accuracy'],
      unique_test_image_files=len(dataset.records),distinct_test_source_groups=len({r['group_id'] for r in dataset.records}),
      misclassified_samples=sum(not p['correct'] for p in predictions),
      by_source={source:describe(m)|{'image_files':sum(r['source']==source for r in dataset.records),
       'distinct_source_groups':len({r['group_id'] for r in dataset.records if r['source']==source})} for source,m in by_source.items()},
      latency={'device':'Apple MPS','batch_size':1,'warmup_iterations':10,'measured_iterations':100,
       'model_forward_mean_ms':float(np.mean(latencies)),'model_forward_median_ms':float(np.median(latencies)),
       'model_forward_p95_ms':float(np.percentile(latencies,95)),
       'method':'Wall clock with MPS synchronization after each request; excludes image loading and preprocessing'})
    with (run/'test_predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
    failures=[p for p in predictions if not p['correct']]
    (run/'misclassified.json').write_text(json.dumps(failures,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(5.5,4.8));ax.imshow(matrix,cmap='Blues')
    ax.set(xlabel='Predicted orientation',ylabel='Labeled orientation',xticks=[0,1],yticks=[0,1],
           xticklabels=['Upright','Upside down'],yticklabels=['Upright','Upside down'],
           title='ViT-Small Orientation\nHeld-out Test Confusion Matrix')
    for i in range(2):
        for j in range(2):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else 'black',fontsize=16)
    fig.tight_layout();fig.savefig(run/'confusion_matrix.png',dpi=160);plt.close(fig)
    if failures:
        selected=failures[:24];sheet=Image.new('RGB',(1200,((len(selected)+3)//4)*270),'white');draw=ImageDraw.Draw(sheet)
        for i,failure in enumerate(selected):
            with Image.open(failure['path']) as im:
                im=im.convert('RGB')
                if failure['virtual_180']:im=im.transpose(Image.Transpose.ROTATE_180)
                im.thumbnail((290,210));x=(i%4)*300;y=(i//4)*270;sheet.paste(im,(x+(290-im.width)//2,y))
            name=Path(failure['path']).stem[:35]
            draw.text((x+4,y+212),f"{failure['source']}: {name}\nlabel={failure['actual']} predicted={failure['predicted']} conf={failure['confidence']:.4f}\nvirtual_180={failure['virtual_180']}",fill='black')
        sheet.save(run/'misclassifications.png')
    predictor=OrientationPredictor(run/'best.pt');checks=[];inference_latencies=[]
    with Image.open(dataset.records[0]['path']) as original:
        original=original.convert('RGB')
        for _ in range(5):predictor.predict_orientation(original)
        for _ in range(50):
            start=time.perf_counter();predictor.predict_orientation(original);inference_latencies.append((time.perf_counter()-start)*1000)
    for row in dataset.records[:3]:
        with Image.open(row['path']) as original:
            original=original.convert('RGB')
            for flipped in (False,True):
                image=original.transpose(Image.Transpose.ROTATE_180) if flipped else original.copy()
                angle,confidence=predictor.predict_orientation(image);corrected=predictor.correct_orientation(image)
                expected=image.transpose(Image.Transpose.ROTATE_180) if angle==180 else image
                assert corrected.size==image.size and corrected.tobytes()==expected.tobytes()
                checks.append({'path':row['path'],'virtual_180':flipped,'predicted_angle':angle,'confidence':confidence,
                               'original_size':list(image.size),'corrected_size':list(corrected.size),'pixel_exact_requested_transform':True})
    summary['latency'].update(predict_orientation_mean_ms=float(np.mean(inference_latencies)),
       predict_orientation_p95_ms=float(np.percentile(inference_latencies,95)),
       predict_orientation_method='50 in-memory PIL requests after 5 warmups, including preprocessing, transfer, forward and confidence extraction')
    (run/'inference_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    (run/'test_metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
