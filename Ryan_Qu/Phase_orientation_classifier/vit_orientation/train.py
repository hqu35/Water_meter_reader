"""Fine-tune on MPS; select by validation accuracy and leave test untouched."""
import argparse
import csv
import json
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import timm
import torch
from torch.utils.data import DataLoader
from .data import load_dataset, verify_manifest
from .model import build, ARCH


def save_json(path, value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def plot_history(history, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,2,figsize=(11,4))
    for axis, metric in zip(axes, ('loss','accuracy')):
        for split in ('train','val'):
            axis.plot([r['epoch'] for r in history], [r[split+'_'+metric] for r in history],label=split)
        axis.set(xlabel='Epoch',ylabel=metric.title());axis.legend();axis.grid(alpha=.2)
    fig.suptitle('ViT-Small Orientation Classifier — Training and Validation')
    fig.tight_layout();tmp=output/'training_curves.tmp.png';fig.savefig(tmp,dpi=150);plt.close(fig);tmp.replace(output/'training_curves.png')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest');p.add_argument('output');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--epochs',type=int,default=25)
    p.add_argument('--patience',type=int,default=5);p.add_argument('--lr',type=float,default=1e-4)
    a=p.parse_args()
    if not torch.backends.mps.is_available():raise RuntimeError('Apple MPS unavailable; no CPU fallback permitted')
    manifest=json.loads(Path(a.manifest).read_text());verify_manifest(manifest)
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    model=build(True).to('mps')
    loaders={s:DataLoader(load_dataset(manifest,s,model,training=s=='train'),batch_size=a.batch_size,
                         shuffle=s=='train',num_workers=0,generator=torch.Generator().manual_seed(a.seed)) for s in ('train','val')}
    if any(len(loader.dataset)==0 for loader in loaders.values()):raise ValueError('Empty train/validation split')
    counts={s:{'images':len(loader.dataset.records),'orientation_samples':len(loader.dataset)} for s,loader in loaders.items()}
    config=vars(a)|{'architecture':ARCH,'pretrained':True,'device':'mps','optimizer':'AdamW',
      'weight_decay':.01,'betas':[.9,.999],'eps':1e-8,'scheduler':None,'loss':'CrossEntropyLoss',
      'checkpoint_selection':'highest_validation_accuracy_earliest_epoch_on_tie','dataset_counts':counts,
      'normalization':{'mean':model.pretrained_cfg['mean'],'std':model.pretrained_cfg['std']},
      'resize':'aspect_ratio_preserving_224_padding_black','train_photometric_jitter':{'brightness':.15,'contrast':.15},
      'full_finetuning':True,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
      'python':sys.version,'python_executable':sys.executable,'torch':torch.__version__,'timm':timm.__version__,
      'platform':platform.platform(),'machine':platform.machine(),'pid':os.getpid()}
    save_json(output/'config.json',config);save_json(output/'split_manifest.json',manifest)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    criterion=torch.nn.CrossEntropyLoss();best=-1;best_epoch=None;wait=0;history=[]
    print('TRAINING_STARTED '+json.dumps(config),flush=True)
    with (output/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['epoch','train_loss','train_accuracy','val_loss','val_accuracy']);writer.writeheader();f.flush()
        for epoch in range(1,a.epochs+1):
            metrics={'epoch':epoch}
            for split,loader in loaders.items():
                model.train(split=='train');loss_sum=correct=count=0
                for step,(x,y) in enumerate(loader,1):
                    x,y=x.to('mps'),y.to('mps')
                    with torch.set_grad_enabled(split=='train'):
                        logits=model(x);loss=criterion(logits,y)
                        if not torch.isfinite(loss).item():raise RuntimeError('Non-finite loss')
                        if split=='train':
                            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                    loss_sum+=loss.item()*len(y);correct+=(logits.argmax(1)==y).sum().item();count+=len(y)
                    if step==1 or step%25==0 or step==len(loader):
                        progress={'epoch':epoch,'phase':split,'batch':step,'batches':len(loader),'loss':loss_sum/count,'accuracy':correct/count}
                        save_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
                metrics[split+'_loss']=loss_sum/count;metrics[split+'_accuracy']=correct/count
            writer.writerow(metrics);f.flush();history.append(metrics);plot_history(history,output)
            print('EPOCH_FINISHED '+json.dumps(metrics),flush=True)
            if metrics['val_accuracy']>best:
                best=metrics['val_accuracy'];best_epoch=epoch;wait=0
                temp=output/'best.tmp.pt';torch.save({'model':model.state_dict(),'config':config,'epoch':epoch,'val_accuracy':best},temp);temp.replace(output/'best.pt')
                print(f'BEST_CHECKPOINT epoch={epoch} val_accuracy={best:.6f}',flush=True)
            else:
                wait+=1
                if wait>=a.patience:
                    print('EARLY_STOPPING',flush=True);break
    result={'status':'finished','epochs_completed':len(history),'best_epoch':best_epoch,'best_validation_accuracy':best,'test_evaluated':False}
    save_json(output/'training_summary.json',result);print('TRAINING_FINISHED '+json.dumps(result),flush=True)
if __name__=='__main__':main()
