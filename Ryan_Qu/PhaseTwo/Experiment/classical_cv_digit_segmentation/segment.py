#!/usr/bin/env python3
"""Exploratory five-wheel geometry from classical image features; no ML models."""
import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d


def preprocess(image, work_width):
    h0, w0 = image.shape[:2]
    w = work_width
    h = max(20, round(h0 * w / w0))
    resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC if w > w0 else cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    # Dark response relative to a broad local background, plus long vertical structures.
    background = cv2.GaussianBlur(smooth, (0, 0), sigmaX=max(4, w / 40), sigmaY=max(2, h / 10))
    dark = np.maximum(background.astype(np.float32) - smooth, 0)
    vertical = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(h * .30)))))
    gx = np.abs(cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3))
    return resized, gray, smooth, dark, vertical, gx


def window_limits(gray):
    h, w = gray.shape
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    profile = gaussian_filter1d(np.median(gy[:, int(.1*w):int(.9*w)], axis=1), 1.2)
    def pick(a, b, fallback):
        if b-a < 3: return fallback, 0.0
        # Edge only trusted above a modest contrast relative to the whole profile.
        section = profile[a:b]; i = a + int(np.argmax(section))
        strength = float(section.max() / (np.median(profile) + 1e-5))
        return (i if strength > 1.15 else fallback), strength
    top, ts = pick(1, max(4, int(.23*h)), 0)
    bottom, bs = pick(min(h-4, int(.77*h)), h-1, h-1)
    if bottom-top < .55*h: return 0, h-1, ts, bs
    return top, bottom, ts, bs


def line_evidence(dark, vertical, gx, top, bottom, slope):
    h, w = dark.shape
    y = np.arange(top, bottom+1)
    x = np.arange(w)
    xx = np.clip(np.rint(x[None,:] + slope*(y[:,None]-(top+bottom)/2)).astype(int), 0, w-1)
    yy = y[:,None]
    # Median resists localized numerals and grime; opening rewards continuity.
    dark_line = np.median(dark[yy, xx], axis=0)
    morph_line = np.mean(vertical[yy, xx], axis=0)
    edge_line = np.median(gx[yy, xx], axis=0)
    def norm(v):
        med=np.median(v); scale=np.percentile(v,90)-med
        return np.clip((v-med)/(scale+1e-5),-1,3)
    raw = .50*norm(dark_line) + .40*norm(morph_line) + .10*norm(edge_line)
    return gaussian_filter1d(raw, 1.2), dark_line, morph_line


def fit_separators(dark, vertical, gx, top, bottom, max_slope, spacing_weight):
    h, w = dark.shape
    # Shared line slope and jointly optimized positions. Unequal spacings remain allowed.
    positions = [np.arange(int(w*(j/5-.085)), int(w*(j/5+.085))+1, 2) for j in range(1,5)]
    best = None
    for slope in np.linspace(-max_slope, max_slope, 17):
        evidence, projection, morph = line_evidence(dark,vertical,gx,top,bottom,float(slope))
        p0=positions[0]
        values=evidence[p0] - spacing_weight*((p0/w-.20)/.15)**2
        back=[]
        for j in range(1,4):
            prev, cur=positions[j-1],positions[j]
            gaps=(cur[:,None]-prev[None,:])/w
            penalty=spacing_weight*((gaps-.20)/.08)**2
            penalty[(gaps<.11)|(gaps>.29)]=1e6
            scores=values[None,:]-penalty
            choice=scores.argmax(axis=1)
            values=evidence[cur]+scores[np.arange(len(cur)),choice]
            back.append(choice)
        values -= spacing_weight*((positions[3]/w-.80)/.15)**2
        k=int(values.argmax()); xs=[int(positions[3][k])]
        for j in (2,1,0):
            k=int(back[j][k]); xs.append(int(positions[j][k]))
        xs=xs[::-1]; objective=float(values.max())
        if best is None or objective>best['objective']:
            best={'x_mid':xs,'slope':float(slope),'objective':objective,'evidence':evidence,'projection':projection,'morph':morph}
    return best


def estimate_widths(fit, width):
    ev=fit['evidence'];xs=fit['x_mid'];result=[]
    for x in xs:
        # Estimate local peak extent, bounded to avoid consuming a whole numeral.
        lo=max(0,x-round(width*.03));hi=min(width-1,x+round(width*.03))
        baseline=float(np.median(np.r_[ev[max(0,x-20):max(1,x-8)],ev[min(width-1,x+8):min(width,x+20)]]))
        peak=float(ev[x]); threshold=baseline+.40*max(peak-baseline,0)
        left=right=x
        while left>lo and ev[left-1]>=threshold:left-=1
        while right<hi and ev[right+1]>=threshold:right+=1
        # One-pixel exclusion is a provisional guard when no line width is evident.
        result.append({'left':left,'right':right,'support':peak,'prominence':peak-baseline,'width_px':right-left+1,'width_provisional':peak-baseline<.35})
    return result


def reconstruct(image, fit, widths, top, bottom):
    h,w=image.shape[:2];s=fit['slope'];mid=(top+bottom)/2
    # Outer and horizontal bounds are window estimates, not verified object edges.
    segments=[(0,widths[0]['left']-1)]
    for a,b in zip(widths,widths[1:]):segments.append((a['right']+1,b['left']-1))
    segments.append((widths[-1]['right']+1,w-1))
    boxes=[]
    for left,right in segments:
        contour=np.array([[left+s*(top-mid),top],[right+s*(top-mid),top],[right+s*(bottom-mid),bottom],[left+s*(bottom-mid),bottom]],np.float32)
        rect=cv2.boxPoints(cv2.minAreaRect(contour))
        boxes.append({'strip_contour':contour.tolist(),'corners':rect.tolist(),'width_mid':right-left,'provisional':True})
    return boxes


def validate(boxes,w,h):
    valid=True; inside=True; overlap=False
    for b in boxes:
        p=np.array(b['corners'],np.float32);area=abs(cv2.contourArea(p))
        if area<4 or b['width_mid']<3: valid=False
        if (p[:,0]<-1e-3).any() or (p[:,0]>w-1+1e-3).any() or (p[:,1]<-1e-3).any() or (p[:,1]>h-1+1e-3).any():inside=False
    for a,b in zip(boxes,boxes[1:]):
        area,_=cv2.intersectConvexConvex(cv2.convexHull(np.array(a['corners'],np.float32)),cv2.convexHull(np.array(b['corners'],np.float32)))
        if area>1e-3:overlap=True
    return {'five_nondegenerate':valid,'nonoverlapping':not overlap,'inside_image':inside,'five_valid_in_image_nonoverlapping':valid and inside and not overlap}


def visualizations(rgb,gray,dark,vertical,fit,widths,top,bottom,boxes):
    h,w=gray.shape; mid=(top+bottom)/2; lines=rgb.copy(); overlay=rgb.copy()
    for x in fit['x_mid']:
        p=(int(round(x+fit['slope']*(top-mid))),top);q=(int(round(x+fit['slope']*(bottom-mid))),bottom)
        cv2.line(lines,p,q,(255,48,0),2,cv2.LINE_AA)
    for y in (top,bottom):cv2.line(lines,(0,y),(w-1,y),(0,210,220),1)
    colors=[(255,40,0),(255,155,0),(40,210,0),(0,165,255),(185,30,245)]
    for box,color in zip(boxes,colors):cv2.polylines(overlay,[np.round(box['corners']).astype(np.int32)],True,color,2,cv2.LINE_AA)
    def disp(a):return cv2.cvtColor(cv2.normalize(a,None,0,255,cv2.NORM_MINMAX).astype(np.uint8),cv2.COLOR_GRAY2RGB)
    diagnostics=np.vstack([np.hstack([disp(gray),disp(dark)]),np.hstack([disp(vertical),disp(np.tile(fit['evidence'][None,:],(h,1)))])])
    return lines,overlay,diagnostics


def yolo_order(points):
    pts=np.array(points);center=pts.mean(axis=0);pts=pts[np.argsort(np.arctan2(pts[:,1]-center[1],pts[:,0]-center[0]))]
    pts=np.roll(pts,-int(np.argmin(pts[:,0]+pts[:,1])),axis=0)
    return pts


def run(args):
    indir=Path(args.input_dir);out=Path(args.output_dir)
    if out.exists() and any(out.iterdir()):raise SystemExit('Output directory is not empty; choose a new path.')
    out.mkdir(parents=True,exist_ok=True);samples=out/'samples';samples.mkdir();records=[]; sheets=[]
    for n in range(args.start,args.end+1):
        matches=[p for p in indir.glob(f'{n}.*') if p.suffix.lower() in ('.png','.jpg','.jpeg')]
        if len(matches)!=1:
            records.append({'id':n,'status':'missing' if not matches else 'ambiguous','matches':[str(p) for p in matches]});continue
        start=time.perf_counter();source=matches[0]
        try:
            original=Image.open(source).convert('RGB');original.load();rgb0=np.asarray(original)
            rgb,gray,smooth,dark,vert,gx=preprocess(rgb0,args.work_width);h,w=gray.shape;top,bottom,ts,bs=window_limits(smooth)
            fit=fit_separators(dark,vert,gx,top,bottom,args.max_slope,args.spacing_weight);widths=estimate_widths(fit,w)
            boxes=reconstruct(rgb,fit,widths,top,bottom);valid=validate(boxes,w,h)
            gaps=np.diff(fit['x_mid'])/w;supports=[a['prominence'] for a in widths]
            plausible=bool(min(supports)>=args.min_prominence and all(.11<=g<=.29 for g in gaps))
            lines,overlay,diag=visualizations(rgb,gray,dark,vert,fit,widths,top,bottom,boxes)
            folder=samples/str(n);folder.mkdir();original.save(folder/'original.png');Image.fromarray(diag).save(folder/'diagnostics.png');Image.fromarray(lines).save(folder/'separators.png');Image.fromarray(overlay).save(folder/'obb.png')
            # Coordinates scale independently to original pixel coordinates.
            sx=rgb0.shape[1]/w;sy=rgb0.shape[0]/h
            orig_boxes=[]
            for box in boxes:
                contour=np.asarray(box['strip_contour'])*[sx,sy]
                corners=cv2.boxPoints(cv2.minAreaRect(contour.astype(np.float32)))
                orig_boxes.append({'contour_xy':contour.tolist(),'obb_xy':corners.tolist(),'provisional':True})
            src_valid=validate([{'corners':b['obb_xy'],'width_mid':np.linalg.norm(np.array(b['contour_xy'][1])-np.array(b['contour_xy'][0]))} for b in orig_boxes],rgb0.shape[1],rgb0.shape[0]);yolo=[]
            all_points=np.concatenate([np.array(b['obb_xy']) for b in orig_boxes]); low=all_points.min(axis=0); high=all_points.max(axis=0)
            pad_left=int(np.ceil(max(0,-low[0])))+2; pad_top=int(np.ceil(max(0,-low[1])))+2
            pad_right=int(np.ceil(max(0,high[0]-(rgb0.shape[1]-1))))+2;pad_bottom=int(np.ceil(max(0,high[1]-(rgb0.shape[0]-1))))+2
            padded=cv2.copyMakeBorder(rgb0,pad_top,pad_bottom,pad_left,pad_right,cv2.BORDER_CONSTANT,value=(255,255,255));Image.fromarray(padded).save(folder/'candidate_padded.png')
            for b in orig_boxes:
                pts=(yolo_order(b['obb_xy'])+[pad_left,pad_top])/[padded.shape[1],padded.shape[0]]
                if not np.all((pts>=0)&(pts<=1)):raise ValueError('padded candidate remains out of bounds')
                yolo.append('0 '+' '.join(f'{v:.8f}' for v in pts.flatten()))
            (folder/'candidate_yolo_obb.txt').write_text('\n'.join(yolo)+'\n')
            elapsed=time.perf_counter()-start
            record={'id':n,'status':'processed','source':str(source),'original_dimensions_wh':[rgb0.shape[1],rgb0.shape[0]],'work_dimensions_wh':[w,h], 'quality':{'four_plausible_separators_heuristic':plausible,'separator_prominences':supports,'min_prominence_threshold':args.min_prominence,'normalized_internal_gaps':gaps.tolist(),'window_top_edge_strength':ts,'window_bottom_edge_strength':bs,'separator_widths_provisional':any(a['width_provisional'] for a in widths),'all_obbs_provisional':True,**src_valid,'exported_in_bounds_obb_count':len(yolo),'yolo_reference_image':'candidate_padded.png','padding_ltrb':[pad_left,pad_top,pad_right,pad_bottom]}, 'geometry':{'line_equation':'x(y) = x_mid + slope * (y - (top + bottom)/2), in original pixel coordinates','separator_x_mid':[x*sx for x in fit['x_mid']],'slope':fit['slope']*sx/sy,'top_y':top*sy,'bottom_y':bottom*sy,'estimated_separator_widths':[dict(a,left=a['left']*sx,right=a['right']*sx,width_px=a['width_px']*sx) for a in widths],'outer_boundaries':'input crop left/right edges; provisional','boxes':orig_boxes},'runtime_seconds':elapsed}
            (folder/'result.json').write_text(json.dumps(record,indent=2)+'\n');records.append(record);sheets.append((n,original,Image.fromarray(overlay),plausible,src_valid))
            print(f'{n}: processed, plausible={plausible}, valid={src_valid["five_valid_in_image_nonoverlapping"]}, {elapsed:.2f}s',flush=True)
        except Exception as exc:
            records.append({'id':n,'status':'failed','error':f'{type(exc).__name__}: {exc}'});print(n,'FAILED',type(exc).__name__,str(exc),flush=True)
    contact_sheet(sheets,out/'contact_sheet.png')
    result={'input_dir':str(indir),'count_requested':args.end-args.start+1,'processed':sum(r['status']=='processed' for r in records),'heuristic_plausible':sum(r.get('quality',{}).get('four_plausible_separators_heuristic',False) for r in records),'five_valid_in_image_nonoverlapping':sum(r.get('quality',{}).get('five_valid_in_image_nonoverlapping',False) for r in records),'parameters':vars(args),'results':records}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


def contact_sheet(items,path):
    # Large enough to inspect all 21; source and overlay preserve aspect ratio.
    row=140;col=540;sheet=Image.new('RGB',(col*2+30,row*(len(items)+1)),(247,249,252));d=ImageDraw.Draw(sheet)
    d.text((75,18),'ORIGINAL',fill=(10,30,50));d.text((col+75,18),'PREDICTED OBB (PROVISIONAL)',fill=(10,30,50))
    for i,(n,original,overlay,plausible,valid) in enumerate(items):
        y=(i+1)*row;d.text((8,y+8),str(n),fill=(0,0,0))
        for x,im in ((50,original),(col+50,overlay)):
            im=im.copy();im.thumbnail((col-70,105),Image.Resampling.LANCZOS);sheet.paste(im,(x,y+4))
        d.text((col+50,y+112),f'line-support: {"yes" if plausible else "weak"} | OBB geometry: {"valid" if valid["five_valid_in_image_nonoverlapping"] else "flagged"}',fill=(30,50,65))
    sheet.save(path)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-dir',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--start',type=int,default=290);p.add_argument('--end',type=int,default=310);p.add_argument('--work-width',type=int,default=400);p.add_argument('--max-slope',type=float,default=.22);p.add_argument('--spacing-weight',type=float,default=.6);p.add_argument('--min-prominence',type=float,default=.25);run(p.parse_args())
