from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
for _d in ('src', 'experiments', 'scripts'):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import json, sys
from pathlib import Path
import numpy as np

root=Path(r'''.''')
sys.path.insert(0,str(root/'data/v2/src'))
from core import cluster_bootstrap_ci

result_path=root/'results/detector/salad_detector_results.json'
pred_path=root/'results/detector/primary_predictions.npz'
data=np.load(pred_path,allow_pickle=True)
result=json.loads(result_path.read_text())
threshold=float(result['tuning']['PITSA_FullTrain']['threshold'])
bootstrap=cluster_bootstrap_ci(data['label'],data['PITSA_FullTrain_clean_scores'],threshold,data['cluster_id'],1000,20261014)
result['pitsa_full_cluster_bootstrap']=bootstrap
result_path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(bootstrap,indent=2,sort_keys=True))
