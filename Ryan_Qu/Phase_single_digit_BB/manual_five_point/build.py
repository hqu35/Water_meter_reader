#!/usr/bin/env python3
"""Build one offline HTML file containing the approved images and metadata."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DEFAULT_MANIFEST = PROJECT/'Phase_orientation_classifier/dataset_2000/manifest.json'


def build(manifest_path, output):
    manifest_path=Path(manifest_path).resolve(); manifest=json.loads(manifest_path.read_text())
    if manifest.get('schema')!='orientation-dataset-v1':raise ValueError('Unsupported input manifest')
    result=[];excluded=[];errors=[];digest=hashlib.sha256()
    for row in manifest['records']:
        if row.get('dataset_group') not in ('clear','clear_one_noise'):continue
        image_id=row.get('original_id')
        if row.get('user_excluded'):
            excluded.append(image_id);continue
        relative=row.get('packaged_image_path');path=(manifest_path.parent/(relative or '')).resolve()
        if not relative or not path.is_relative_to(manifest_path.parent) or not path.is_file():errors.append(f'{image_id}: missing or unsafe path');continue
        label=row.get('orientation_label');evidence=row.get('label_evidence')
        if (label,evidence) not in [('upright','unmarked_assumed_upright'),('upside_down','explicit_upside_down_mark')]:errors.append(f'{image_id}: missing or ambiguous orientation');continue
        flag=row.get('author_scenario_labels',{}).get('six_digit')
        if type(flag) is not int or flag not in (0,1):errors.append(f'{image_id}: missing author six_digit flag');continue
        content=path.read_bytes();sha=hashlib.sha256(content).hexdigest()
        if sha!=row.get('sha256'):errors.append(f'{image_id}: image checksum mismatch');continue
        with Image.open(io.BytesIO(content)) as im:
            if im.format!='PNG':errors.append(f'{image_id}: expected a PNG');continue
            width,height=im.size;im.verify()
        digest.update((image_id+'\0'+sha+'\0'+label+'\0'+str(flag)+'\0').encode())
        result.append({'image_id':image_id,'source_image':relative,'sha256':sha,'dataset_group':row['dataset_group'],
                       'width':width,'height':height,'orientation':label,
                       'orientation_evidence':'user_confirmed_unmarked_upright_2026-11-09' if label=='upright' else evidence,
                       'six_digit':flag,'digit_count':6 if flag else 5,'src':'data:image/png;base64,'+base64.b64encode(content).decode('ascii')})
    if errors:raise ValueError('Dataset problems (no output written):\n'+'\n'.join(errors[:30]))
    if len(result)!=1996 or len({x['image_id'] for x in result})!=1996 or len(excluded)!=4:
        raise ValueError(f'Unexpected scope: {len(result)} included, {len(excluded)} excluded; refusing to build.')
    data={'schema':'watermeter-register-crops-v1','fingerprint':digest.hexdigest(),'excluded_original_ids':excluded,'images':result}
    template=(HERE/'template.html').read_text();geometry=(HERE/'geometry.js').read_text()
    assert template.count('__DATASET__')==1 and template.count('__GEOMETRY__')==1 and template.count('__APP__')==1
    html=template.replace('__DATASET__',json.dumps(data,separators=(',',':')).replace('</','<\\/')).replace('__GEOMETRY__',geometry).replace('__APP__',(HERE/'guided_app.js').read_text())
    output=Path(output)
    if output.exists():raise FileExistsError(f'Refusing to overwrite {output}; choose a new path')
    output.write_text(html)
    print(f'Built {output} with {len(result)} images and {len(excluded)} exclusions; fingerprint {digest.hexdigest()[:12]}')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST);p.add_argument('--output',type=Path,default=HERE/'index.html');args=p.parse_args();build(args.manifest,args.output)
