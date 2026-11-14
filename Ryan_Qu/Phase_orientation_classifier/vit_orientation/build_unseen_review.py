"""Build a portable ten-column human review of unseen orientation predictions."""
import argparse,base64,collections,hashlib,io,json
from pathlib import Path
from PIL import Image


def main():
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args()
 source=json.loads((args.directory/'predictions.json').read_text());items=[];h=hashlib.sha256()
 for r in source['records']:
  item={'id':r['id'],'index':r['index'],'predicted_label':r.get('predicted_label'),
    'confidence':r.get('confidence'),'status':'success' if r['status']=='success' else 'failed',
    'initially_excluded':False,'image_sha256':r.get('rectified_sha256'),'original_sha256':r['original_sha256']}
  if r['status']=='success':
   path=args.directory/r['rectified_image_path']
   assert hashlib.sha256(path.read_bytes()).hexdigest()==r['rectified_sha256']
   with Image.open(path) as im:
    im=im.convert('RGB');im.thumbnail((900,900));buffer=io.BytesIO();im.save(buffer,format='WEBP',quality=87,method=4)
   item['preview']='data:image/webp;base64,'+base64.b64encode(buffer.getvalue()).decode()
  h.update(json.dumps({k:v for k,v in item.items() if k!='preview'},sort_keys=True).encode());items.append(item)
 template=Path(__file__).with_name('label_review.html').read_text()
 replacements=[
  ('Water Meter Orientation Label Review','Unseen Water Meter Orientation — 100-Image Pilot'),
  ("Full geometrically rectified images, <b>before</b> the old classifier's 180° correction. Judge the visible digit reading direction.",
   'Previously unused image IDs: Stage I geometric rectification → trained ViT-Small. Displayed images have <b>not</b> been flipped by the classifier.'),
  ('0 = Upright · 1 = Upside down · Each row contains 10 images. Click a picture to enlarge it.',
   '0 = Normal · 1 = Upside down · 10 images per row. Click to enlarge. Select the actual visible orientation to confirm or correct the prediction.'),
  ('Upright','Normal'),('upright','normal'),('restored_label','predicted_label'),
  ('Restored:','Prediction:'),('Restored: ','Prediction: '),
  ('restored-orientation-review-v1','unseen-orientation-review-v1'),
  ('watermeter_orientation_label_review.json','watermeter_unseen100_orientation_review.json'),
  ('excluded_by_previous_review','initially_excluded'),
  ('This image was previously rejected for crop or information quality. Restore it to the labeled set?',
   'Restore this excluded image to the labeled set?'),
  ('Previously rejected; excluded from training.','Excluded from this review.'),
  ("restored.textContent='Prediction: '+labelName(item.predicted_label)",
   "restored.textContent='Prediction: '+labelName(item.predicted_label)+(item.confidence==null?'':' · score '+(item.confidence*100).toFixed(1)+'%')"),
  ("image_sha256:item.image_sha256,predicted_label:item.predicted_label", "image_sha256:item.image_sha256,original_sha256:item.original_sha256,confidence:item.confidence,predicted_label:item.predicted_label"),
  ("' · Prediction: '","' · Prediction: '"),
 ]
 for old,new in replacements:
  template=template.replace(old,new)
 # The viewer uses the same immutable prediction as its card.
 template=template.replace("' · Restored: '","' · Prediction: '")
 data={'schema':'watermeter-unseen-orientation-dataset-v1','fingerprint':h.hexdigest(),
       'label_stage':'full_stage1_geometrically_rectified_image_before_classifier_correction',
       'config':source['config'],'images':items}
 assert len({r['id'] for r in items})==len(items)==100
 template=template.replace('__DATASET__',json.dumps(data,separators=(',',':')).replace('</','<\\/'))
 (args.directory/'index.html').write_text(template)
 (args.directory/'review_dataset.json').write_text(json.dumps({k:v for k,v in data.items() if k!='images'}|{'images':[{k:v for k,v in i.items() if k!='preview'} for i in items]},indent=2)+'\n')
 print('Built',args.directory/'index.html','with',len(items),'fixed samples')
if __name__=='__main__':main()
