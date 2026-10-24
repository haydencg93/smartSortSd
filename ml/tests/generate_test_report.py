"""Parse actual pytest report and summarize requirement-level integration status."""
import json
import subprocess
from pathlib import Path
from datetime import datetime,timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
cmd=['python','-m','pytest',str(ROOT/'tests'),'-q','--disable-warnings','--tb=short']
p=subprocess.run(cmd,capture_output=True,text=True,cwd=ROOT)
lines=(p.stdout+'\n'+p.stderr).strip()
(ROOT/'results'/'pytest_results.txt').write_text(lines+'\n')
print(lines)
assert p.returncode==0, 'Some automated tests failed; see results/pytest_results.txt'
import re
m=re.search(r'(\d+) passed',lines)
n=int(m.group(1)) if m else 0
checks=[
  ('Checkpoint architecture','PASS','Loaded, correct 6-class output tensor'),
  ('Model consistency','PASS','Deterministic predictions and preprocessing match'),
  ('FastAPI functionality','PASS','Health, uploads, schema, invalid inputs, concurrency'),
  ('PC presence helpers','PASS','Difference mask and hysteresis parameters'),
  ('Evaluation and chart generator','PASS','Metric formula and image outputs checked with labeled fixtures'),
  ('External photographic accuracy','PENDING','Independent labeled photographs not accessible here'),
  ('Iowa City item-level routing','PENDING','Independent per-item accept/reject labels required'),
  ('End-to-end 5 second system','PENDING','Physical capture/display not tested'),
  ('Barcode and LCD','NOT IN UPLOAD','No implemented barcode or LCD path in provided files'),
  ('Timestamped event storage','NOT IN UPLOAD','Not implemented in uploaded app/client files'),
  ('Camera proximity gating','NOT IN UPLOAD','PC image differencing is not a 24-inch proximity sensor'),
]
report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'tests_passed':n,'return_code':p.returncode,
        'categories':[dict(name=a,status=b,evidence=c) for a,b,c in checks],
        'limitations':'PASS indicates PC functional behavior only. Pending checks cannot be inferred from synthetic images.'}
(ROOT/'results'/'functional_verification.json').write_text(json.dumps(report,indent=2))

status_colors={'PASS':'#208b55','PENDING':'#e2a33d','NOT IN UPLOAD':'#a4646f'}
fig,ax=plt.subplots(figsize=(13.6,10))
ax.set_xlim(0,12);ax.set_ylim(-.8,len(checks)+.4);ax.invert_yaxis();ax.axis('off')
for i,(name,status,desc) in enumerate(checks):
    ax.add_patch(plt.Rectangle((0,i-.38),12,.87,color=('#eef4f3' if i%2==0 else '#f8fafb')))
    ax.text(.25,i-.11,name,ha='left',va='center',fontsize=12,fontweight='bold',color='#233239')
    ax.text(.25,i+.23,desc,ha='left',va='center',fontsize=9,color='#55656c')
    ax.text(10.2,i+.02,status,ha='center',va='center',fontsize=10,fontweight='bold',color='white',
            bbox={'facecolor':status_colors[status],'edgecolor':'none','boxstyle':'round,pad=.45'})
ax.set_title('Smart Sort · test coverage and requirement audit',fontsize=20,pad=24,color='#24383d')
fig.text(.08,.035,f'{n} automated tests passed. Pending and not-implemented requirements are NOT passes.',fontsize=11,color='#435660')
fig.savefig(ROOT/'results'/'verification_matrix.png',dpi=180,bbox_inches='tight',facecolor='white')
plt.close()
print('Verification matrix and results generated')
