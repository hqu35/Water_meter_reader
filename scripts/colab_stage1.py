"""Local CLI bridge. Never provisions, stops, or restarts a Colab runtime.

python scripts/colab_stage1.py status|sync|PHASE|fetch
PHASE: gpu download inspect prepare approve train evaluate package
"""
from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess, sys, zipfile

REPO=Path(__file__).resolve().parents[1]
REMOTE='/content/cs539_stage1'
SESSION='cs539-stage1'

def cli(*args,stdin=None):
 binary=shutil.which('colab') or str(Path.home()/'.local/bin/colab')
 return subprocess.run([binary,*args],input=stdin,text=True,check=True,cwd=REPO)

def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['status','sync','push-prepared','gpu','download','inspect','prepare','approve','train','normalize-checkpoints','evaluate','package','fetch']);p.add_argument('--reviewer');p.add_argument('--background',action='store_true');a=p.parse_args()
 output=REPO/'outputs/stage1';output.mkdir(parents=True,exist_ok=True)
 if a.phase=='status':cli('status','--session',SESSION);return
 if a.phase=='sync':
  cli('exec','--session',SESSION,stdin=f"from pathlib import Path\np=Path('{REMOTE}/project')\n(p/'scripts').mkdir(parents=True,exist_ok=True)\n(p/'configs').mkdir(exist_ok=True)\n")
  for f in ['scripts/stage1.py','configs/stage1.json','requirements-stage1.txt']:
   cli('upload','--session',SESSION,str(REPO/f),REMOTE+'/project/'+f)
  return
 if a.phase=='push-prepared':
  # Browser-downloaded archive may be prepared locally. Transfer only the selected
  # real subset in small CLI uploads, avoiding a multi-GB base64 allocation.
  assert (output/'artifacts/annotation_review.json').exists(),'Review locally before transfer'
  bundle=output/'prepared_subset.zip'
  with zipfile.ZipFile(bundle,'w',zipfile.ZIP_STORED) as z:
   for folder in ['prepared','artifacts']:
    for f in sorted((output/folder).rglob('*')):
     if f.is_file():z.write(f,str(f.relative_to(output)))
  chunks=[]
  with bundle.open('rb') as f:
   while True:
    b=f.read(32*1024*1024)
    if not b:break
    part=output/f'subset.part{len(chunks):03d}';part.write_bytes(b)
    chunks.append(dict(name=part.name,sha256=hashlib.sha256(b).hexdigest()))
    cli('upload','--session',SESSION,str(part),REMOTE+'/'+part.name)
  h=hashlib.sha256(bundle.read_bytes()).hexdigest()
  code=f"from pathlib import Path\nimport hashlib,zipfile,yaml,json\nroot=Path('{REMOTE}')\nparts={chunks!r}\n"
  code+="target=root/'prepared_subset.zip'\nwith target.open('wb') as out:\n for part in parts:\n  b=(root/part['name']).read_bytes()\n  assert hashlib.sha256(b).hexdigest()==part['sha256']\n  out.write(b)\n"
  code+=f"assert hashlib.sha256(target.read_bytes()).hexdigest()=={h!r}\n"
  code+="with zipfile.ZipFile(target) as z:\n for n in z.namelist():assert (root/n).resolve().is_relative_to(root.resolve())\n z.extractall(root)\np=root/'prepared/data.yaml'\ndata=yaml.safe_load(p.read_text());data['path']=str(root/'prepared');p.write_text(yaml.safe_dump(data,sort_keys=False))\nprint('Verified prepared subset transferred; raw archive was not uploaded.')\n"
  cli('exec','--session',SESSION,'--timeout','60',stdin=code);return
 if a.phase=='fetch':
  cli('download','--session',SESSION,REMOTE+'/results_bundle.json',str(output/'results_bundle.json'))
  report=json.loads((output/'results_bundle.json').read_text());bundle=output/'stage1_results.zip'
  cli('download','--session',SESSION,REMOTE+'/stage1_results.zip',str(bundle))
  assert hashlib.sha256(bundle.read_bytes()).hexdigest()==report['sha256'],'Retrieved bundle hash mismatch'
  with zipfile.ZipFile(bundle) as z:
   for n in z.namelist():assert (output/n).resolve().is_relative_to(output.resolve()),'Unsafe archive member'
   z.extractall(output)
  checks=json.loads((output/'artifacts/artifact_checksums.json').read_text())
  for f,h in checks.items():assert hashlib.sha256((output/'artifacts'/f).read_bytes()).hexdigest()==h,f
  print('Verified local checkpoints, metrics and figures:',output/'artifacts');return
 if a.phase=='approve' and not a.reviewer:p.error('--reviewer is required after visual inspection of all review sheets')
 # Remote Python executable is deliberately resolved on Colab, not copied from macOS.
 code="import os, subprocess, sys\nfrom pathlib import Path\n"
 code+=f"args=[sys.executable,'{REMOTE}/project/scripts/stage1.py','--root','{REMOTE}',{a.phase!r}]\n"
 if a.reviewer:code+=f"args+=['--reviewer',{a.reviewer!r}]\n"
 if a.background:
  code+=f"log=Path('{REMOTE}/{a.phase}.log').open('w')\np=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd='{REMOTE}/project')\nprint('Started PID',p.pid,'log: {REMOTE}/{a.phase}.log')\n"
 else:code+="subprocess.run(args,check=True)\n"
 cli('exec','--session',SESSION,'--timeout','60',stdin=code)

if __name__=='__main__':main()
