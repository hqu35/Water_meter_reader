"""Apply the Stage 1 checkpoint to a real photograph and save a crop + coordinates."""
import argparse,json,math
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--image',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('outputs/stage1/prediction'));p.add_argument('--device',default='cpu');p.add_argument('--confidence',type=float,default=.25);a=p.parse_args()
 from ultralytics import YOLO
 from PIL import Image
 assert a.weights.is_file() and a.image.is_file()
 a.output.mkdir(parents=True,exist_ok=True)
 result=YOLO(str(a.weights)).predict(source=str(a.image),conf=a.confidence,imgsz=640,device=a.device,verbose=False)[0]
 result.save(filename=str(a.output/'prediction.jpg'))
 image=Image.open(a.image).convert('RGB');boxes=result.boxes.xyxy.cpu().tolist();scores=result.boxes.conf.cpu().tolist();best=max(range(len(scores)),key=lambda i:scores[i]) if scores else None
 record=dict(source=str(a.image),confidence_threshold=a.confidence,boxes_xyxy=boxes,confidences=scores,register_found=best is not None)
 if best is not None:
  x0,y0,x1,y1=boxes[best];crop=[max(0,math.floor(x0)),max(0,math.floor(y0)),min(image.width,math.ceil(x1)),min(image.height,math.ceil(y1))]
  image.crop(crop).save(a.output/'register_crop.png');record.update(selected_box=boxes[best],selected_confidence=scores[best],crop_xyxy=crop)
 (a.output/'prediction.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))

if __name__=='__main__':main()
