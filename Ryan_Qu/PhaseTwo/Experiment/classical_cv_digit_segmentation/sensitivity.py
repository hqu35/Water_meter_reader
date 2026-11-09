#!/usr/bin/env python3
"""Deterministic perturbation stability, not accuracy or ground-truth evaluation."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from segment import preprocess,window_limits,fit_separators

def detect(image):
 _,gray,smooth,dark,vert,gx=preprocess(image,400);t,b,_,_=window_limits(smooth);fit=fit_separators(dark,vert,gx,t,b,.22,.6)
 return np.column_stack([np.array(fit['x_mid'])*image.shape[1]/400,np.full(4,(t+b)/2*image.shape[0]/gray.shape[0])])

def main():
 p=argparse.ArgumentParser();p.add_argument('--input-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.input_dir);rows=[]
 for n in [291,302,306]:
  image=np.asarray(Image.open(root/f'{n}.png').convert('RGB'));h,w=image.shape[:2];base=detect(image)
  operations={'blur_sigma_1.5':(cv2.GaussianBlur(image,(0,0),1.5),None),'darken_40_percent':(np.uint8(np.round(image*.6)),None)}
  for ang in [-5,5]:
   m=cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2),ang,1);out=cv2.warpAffine(image,m,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE);operations[f'rotation_{ang:+d}_deg']=(out,m)
  for kind,(changed,m) in operations.items():
   q=detect(changed)
   if m is not None:q=np.column_stack([q,np.ones(4)])@cv2.invertAffineTransform(m).T
   diff=np.abs(q[:,0]-base[:,0]);rows.append({'sample':n,'perturbation':kind,'mean_boundary_shift_px':float(diff.mean()),'mean_boundary_shift_percent_width':float(100*diff.mean()/w),'individual_shift_px':diff.tolist()})
 Path(a.output).write_text(json.dumps({'disclaimer':'Shift of fitted line midpoints relative to unperturbed prediction; not accuracy. Rotation compared after inverse mapping; same-size border-replicated canvas.','results':rows},indent=2)+'\n')
 for row in rows:print(row['sample'],row['perturbation'],round(row['mean_boundary_shift_percent_width'],2),'% width')
if __name__=='__main__':main()
