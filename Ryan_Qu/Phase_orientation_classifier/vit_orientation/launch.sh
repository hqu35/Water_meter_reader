#!/bin/bash
# Launch a single training run in the background. Run from Phase_orientation_classifier.
set -euo pipefail
if [[ $# -ne 2 ]]; then echo 'Usage: launch.sh SPLIT_MANIFEST RUN_NAME' >&2; exit 2; fi
manifest="$1"; name="$2"; python_bin="${PYTHON_BIN:-python3}"
[[ "$name" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid run name' >&2; exit 2; }
[[ -f "$manifest" ]] || { echo 'Missing split manifest' >&2; exit 2; }
mkdir -p outputs/logs
[[ ! -e "outputs/$name" && ! -e "outputs/logs/$name.log" ]] || { echo 'Run or log already exists' >&2; exit 2; }
"$python_bin" -c 'import torch; assert torch.backends.mps.is_available(), "MPS unavailable"'
"$python_bin" -c 'import json,sys; from vit_orientation.data import verify_manifest; verify_manifest(json.load(open(sys.argv[1])))' "$manifest"
export HF_HOME="$PWD/outputs/huggingface"
"$python_bin" - "$manifest" "$name" <<'PYLAUNCH'
import json,os,subprocess,sys
from pathlib import Path
manifest,name=sys.argv[1:]
log=Path('outputs/logs')/(name+'.log')
command=[sys.executable,'-u','-m','vit_orientation.train',manifest,'outputs/'+name]
with log.open('xb') as f:
    process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,
                             start_new_session=True,close_fds=True)
info={'pid':process.pid,'command':command,'cwd':str(Path.cwd()),'log':str(log.resolve()),
      'output':str((Path('outputs')/name).resolve())}
Path('outputs/logs',name+'.launch.json').write_text(json.dumps(info,indent=2)+'\n')
print(json.dumps(info,indent=2))
PYLAUNCH
