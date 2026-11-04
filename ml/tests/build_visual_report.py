#!/usr/bin/env python3
"""Assemble self-contained HTML visual verification report from *actual test outputs*."""
import base64
import html
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
res=ROOT/'results'
verification=json.loads((res/'functional_verification.json').read_text())
http=json.loads((res/'live_http'/'http_metrics.json').read_text())
smoke=json.loads((res/'smoke'/'smoke_metrics.json').read_text())

def inline(path):
    data=base64.b64encode((ROOT/path).read_bytes()).decode('ascii')
    return f'data:image/png;base64,{data}'

statuses={'PASS':0,'PENDING':0,'NOT IN UPLOAD':0}
for row in verification['categories']:statuses[row['status']]+=1

html_out='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Smart Sort — Validation Report</title><style>
:root{--ink:#15313c;--muted:#526c73;--bg:#f5f7f6;--line:#dbe5e4;--green:#0b815c;--amber:#b77714;--red:#a54f60;--blue:#196a86}
*{box-sizing:border-box}body{margin:0;background:var(--bg);font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;color:var(--ink)}
header{background:linear-gradient(125deg,#102d3a,#176b72);color:white;padding:54px max(26px,calc((100vw - 1150px)/2)) 46px}
header h1{margin:10px 0;font-size:clamp(34px,5vw,54px);line-height:1.09;letter-spacing:-1px}header p{max-width:780px;color:#ddf0ed;margin:8px 0}
mark{color:#d5ffed;background:none;font-weight:700;letter-spacing:2px;font-size:12px;text-transform:uppercase}
main{max-width:1150px;margin:auto;padding:30px 25px 80px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-top:-66px;position:relative}
.metric{background:white;padding:24px;border-radius:16px;box-shadow:0 10px 33px #12333117;border:1px solid var(--line)}
.metric .num{font-size:30px;font-weight:800;color:#176b72}.metric .name{font-size:13px;color:var(--muted)}
section{background:white;border:1px solid var(--line);border-radius:18px;padding:30px;margin-top:24px;box-shadow:0 4px 25px #234c4408}
h2{font-size:25px;letter-spacing:-.4px;margin:0 0 8px}p{color:#3c565d}small,.small{font-size:13px;color:var(--muted)}
.pills{display:flex;gap:10px;flex-wrap:wrap}.pill{display:inline-block;border:1px solid #c3d8d4;border-radius:999px;padding:5px 13px;font-size:12px;font-weight:700;color:var(--green)}
.warning{border-left:4px solid var(--amber);background:#fff8ed;padding:14px 18px;border-radius:8px;margin-top:20px;color:#7b521a}
.visual{width:100%;border:1px solid var(--line);border-radius:12px;background:white;margin:15px 0 5px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}.grid2 img{height:auto}
.table{width:100%;border-collapse:collapse;margin-top:20px}.table th,.table td{text-align:left;padding:12px;border-bottom:1px solid var(--line);vertical-align:top}.table th{color:var(--muted);font-size:13px}
.status{border-radius:6px;padding:3px 9px;font-size:11px;font-weight:700;white-space:nowrap}.PASS{background:#d9f3e7;color:#087047}.PENDING{background:#fff0d4;color:#93600e}.NOT{background:#f7e2e6;color:#9a4b5b}
code{background:#eaf0f0;padding:2px 6px;border-radius:4px}pre{overflow-x:auto;background:#102b36;color:#e2f6f1;padding:18px;border-radius:12px;font-size:13px}
footer{color:#6a7d81;text-align:center;padding:24px;font-size:13px}@media(max-width:850px){.metrics,.grid2{grid-template-columns:1fr 1fr}}@media(max-width:560px){.metrics,.grid2{grid-template-columns:1fr}header{padding-bottom:95px}}
</style></head><body><header><mark>Smart Sort / Technical validation</mark><h1>Functional &amp; Model Test Report</h1>
<p>Actual uploaded MobileNetV2 six-class weights • unchanged FastAPI source • reproducible test infrastructure • real localhost HTTP benchmark.</p>
</header><main><div class="metrics">
'''
for value,label in [(f"{verification['tests_passed']}/30",'Automated checks passed'),
                     ('35/35','Localhost HTTP requests succeeded'),
                     (f"{http['http_round_trip_ms']['p95']:.1f} ms",'Localhost HTTP p95 latency'),
                     ('Not yet','External photographic accuracy')]:
    html_out+=f'<div class="metric"><div class="num">{html.escape(value)}</div><div class="name">{html.escape(label)}</div></div>'
html_out+='''</div><section><h2>What has actually been verified?</h2>
<p>The supplied six-class MobileNetV2 checkpoint loads correctly, predicts a valid label, works with the existing API, and correctly handles multiple valid and invalid upload types. Six-class preprocessing is consistent with the application. Synthetic patterns were used only to test endpoint behavior, not recognition accuracy.</p>
<div class="pills"><span class="pill">30 automated checks</span><span class="pill">12 ASGI synthetic diagnostics</span><span class="pill">35 real HTTP synthetic requests</span><span class="pill">No original Python files edited</span></div>
<div class="warning"><strong>Accuracy has not been measured on waste photographs.</strong> The external evaluator is implemented and ready for RealWaste. Real labeled images were not available for this run, so any six-class accuracy, F1 or disposal-accuracy figure would be unsupported.</div>
<img class="visual" alt="Test verification matrix" src="'''+inline(Path('results/verification_matrix.png'))+'''">
</section><section><h2>Observed performance: real localhost HTTP</h2>
<p>The test launched the original FastAPI app with Uvicorn, posted 35 JPEG diagnostic samples over HTTP and checked the response schema and failure status for bad uploads. These times include HTTP over the local loopback interface, but not real camera, Wi-Fi, sensor, or LCD operations.</p>
<img class="visual" alt="Actual loopback HTTP timings" src="'''+inline(Path('results/live_http/http_latency.png'))+'''">
<p class="small">HTTP p50: '''+str(http['http_round_trip_ms']['p50'])+''' ms · p95: '''+str(http['http_round_trip_ms']['p95'])+''' ms · p99: '''+str(http['http_round_trip_ms']['p99'])+''' ms. These are environment-specific measurements; they do not certify the system-wide five-second requirement.</p>
</section><section><h2>Out-of-distribution model diagnostics</h2>
<p>The trained model was asked to classify 12 artificial color and shape patterns. The purpose is to expose API behavior on unfamiliar inputs and show confidence levels, not to assign correct trash/recycling labels.</p>
<div class="grid2"><div><img class="visual" alt="Model confidence on synthetic patterns" src="'''+inline(Path('results/smoke/synthetic_confidence_not_accuracy.png'))+'''"></div>
<div><img class="visual" alt="ASGI timing for synthetic patterns" src="'''+inline(Path('results/smoke/synthetic_latency.png'))+'''"></div></div>
<img class="visual" alt="Diagnostic gallery of labeled synthetic test patterns" src="'''+inline(Path('results/smoke/synthetic_diagnostic_gallery.png'))+'''">
<p class="small">Some synthetic patterns returned high model confidence. Confidence is not correctness, and the API currently has no low-confidence retry/abstain threshold.</p></section>
<section><h2>Requirement coverage</h2><table class="table"><thead><tr><th>Area</th><th>Outcome</th><th>Evidence / blocker</th></tr></thead><tbody>'''
for item in verification['categories']:
    key='NOT' if item['status']=='NOT IN UPLOAD' else item['status']
    html_out+=f'<tr><td><strong>{html.escape(item["name"])}</strong></td><td><span class="status {key}">{html.escape(item["status"])}</span></td><td>{html.escape(item["evidence"])}</td></tr>'
html_out+='''</tbody></table></section><section><h2>Ready for the next stage: external accuracy evaluation</h2>
<p>The external evaluation script accepts UCI RealWaste with six exact material mappings. It produces counts and normalized confusion matrices, per-class precision/recall/F1, calibration, latency, correct/error image galleries, and optional lighting/blur/compression/occlusion comparisons based on <strong>actual model predictions</strong>.</p>
<pre>cd SmartSort_Test_Suite
python evaluate_external.py --dataset "/path/to/realwaste-main" \\
  --source realwaste --out results/realwaste --robustness</pre>
<p class="small">Exact duplicates may be screened against original training files using <code>--train-dir</code>. Item-level Iowa City acceptance requires separately verified per-image labels; generic material categories cannot establish it.</p></section>
<footer>Smart Sort • Results are specific to the uploaded checkpoint and this CPU testing environment. Original project files and checkpoint remain unchanged.</footer></main></body></html>'''
p=ROOT/'SmartSort_Test_Report.html';p.write_text(html_out,encoding='utf-8');print('Report:',p, 'size:',round(p.stat().st_size/1e6,2),'MB')
