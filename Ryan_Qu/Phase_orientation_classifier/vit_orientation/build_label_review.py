"""Build a portable 10-column HTML review of restored full-image orientation labels."""
import argparse,base64,collections,hashlib,io,json
from pathlib import Path
from PIL import Image

def main():
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();records=[];h=hashlib.sha256()
 files=list((a.directory/'records').glob('*.json'))
 if len(files)!=2000:raise ValueError(f'Expected all 2,000 records, found {len(files)}')
 for f in files:records.append(json.loads(f.read_text()))
 records.sort(key=lambda r:r['index']);assert len({r['id'] for r in records})==2000
 images=[]
 for r in records:
  item={k:r.get(k) for k in ['id','index','restored_label','status','excluded_by_previous_review','image_sha256']}
  if r['status']!='failed':
   path=a.directory/r['image_path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['image_sha256']
   with Image.open(path) as im:
    im=im.convert('RGB');im.thumbnail((640,640));b=io.BytesIO();im.save(b,format='WEBP',quality=82,method=4)
   item['preview']='data:image/webp;base64,'+base64.b64encode(b.getvalue()).decode()
  h.update(json.dumps({k:item[k] for k in item if k!='preview'},sort_keys=True).encode());images.append(item)
 data={'schema':'watermeter-restored-orientation-dataset-v1','fingerprint':h.hexdigest(),'label_stage':'geometrically_rectified_full_image_before_legacy_orientation_classifier','images':images}
 template=Path(__file__).with_name('label_review.html').read_text();assert template.count('__DATASET__')==1
 html=template.replace('__DATASET__',json.dumps(data,separators=(',',':')).replace('</','<\\/'))
 tmp=a.directory/'index.building.html';tmp.write_text(html);tmp.replace(a.directory/'index.html')
 (a.directory/'restored_labels.json').write_text(json.dumps({'schema':'watermeter-restored-orientation-v1','label_stage':data['label_stage'],'records':records},indent=2)+'\n')
 print(json.dumps({'count':len(records),'status':dict(collections.Counter(r['status'] for r in records)),'restored_labels':dict(collections.Counter(str(r.get('restored_label')) for r in records)),'html_bytes':len(html.encode())},indent=2))
if __name__=='__main__':main()
