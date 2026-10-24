r"""Rebuild Smart Sort's 100-image HTML report from saved results.csv without inference or Matplotlib.

Run from ml/tests:
  python .\SmartSort_100_Image_Benchmark\generate_report_from_csv.py --results .\results_100 --images .\heldout_neoavara\images
"""
import argparse
import csv
import json
import sys
import html
import base64
from collections import Counter
from pathlib import Path

# Load the original benchmark's report templates without invoking its ML code.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as original

COLORS = original.STATUS_COLORS

def svg_data(svg):
    return 'data:image/svg+xml;base64,' + base64.b64encode(svg.encode('utf-8')).decode('ascii')

def svg_chart(rows, kind):
    W, H = 1280, 380
    if kind == 'pie':
        counts = Counter(r['status'] for r in rows)
        total = max(1, len(rows))
        x = 60
        bits = [f'<rect width="{W}" height="{H}" fill="white"/>', '<text x="45" y="35" font-size="23" font-family="Arial" fill="#172437">Test outcome distribution</text>']
        y = 82
        for status, color in COLORS.items():
            n = counts[status]
            bits.extend([f'<rect x="50" y="{y}" width="22" height="22" fill="{color}"/>', f'<text x="83" y="{y+18}" font-size="20" font-family="Arial" fill="#172437">{html.escape(status)}: {n} ({n/total*100:.1f}%)</text>'])
            y += 52
        bits.append(f'<rect x="50" y="315" width="1100" height="36" rx="8" fill="#e6eaf0"/>')
        offset=50
        for status, color in COLORS.items():
            n=counts[status]
            width=1100*n/total
            if width>0:
                bits.append(f'<rect x="{offset:.2f}" y="315" width="{width:.2f}" height="36" fill="{color}"/>')
            offset+=width
        bits.append('</svg>')
        return svg_data('<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="380" viewBox="0 0 1280 380">'+''.join(bits))

    is_time = kind == 'times'
    vals = [float(r['elapsed_ms']) if is_time else float(r['confidence'])*100 for r in rows]
    # Fit the time axis to actual measurements with exactly 50 ms headroom.
    # The 2.5-second concern rule remains unchanged; no goal is plotted.
    threshold = 75
    maxv = (max(vals, default=0) + 50) if is_time else 100
    left, top, plot_w, plot_h = 75, 76, 1130, 230
    bar_w = plot_w / max(1,len(vals))
    title = 'Processing time for each test image' if is_time else 'Model confidence for each test image'
    axis = 'Milliseconds (preprocessing + inference)' if is_time else 'Confidence (%)'
    y_threshold = top+plot_h - threshold/maxv*plot_h if not is_time else None
    bits = [f'<rect width="{W}" height="{H}" fill="white"/>',
            f'<text x="45" y="35" font-size="23" font-family="Arial" fill="#172437">{title}</text>',
            f'<text x="{left}" y="57" font-size="14" font-family="Arial" fill="#526173">{axis}</text>',
            f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#8995a7"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#8995a7"/>']
    for k in range(5):
        v=maxv*k/4
        y=top+plot_h-k*plot_h/4
        bits.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e6eaf0"/>')
        bits.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="12" font-family="Arial" fill="#526173">{v:.0f}</text>')
    for i,(r,v) in enumerate(zip(rows,vals)):
        h=max(0,v/maxv*plot_h)
        x=left + i*bar_w + bar_w*.1
        y=top+plot_h-h
        color=COLORS.get(r['status'],'#64748b')
        bits.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w*.8:.2f}" height="{h:.2f}" fill="{color}"><title>{html.escape(r["id"])}: {v:.2f}{"ms" if is_time else "%"}</title></rect>')
    if not is_time:
        bits.extend([f'<line x1="{left}" y1="{y_threshold:.2f}" x2="{left+plot_w}" y2="{y_threshold:.2f}" stroke="#e4a600" stroke-width="2" stroke-dasharray="7,6"/>',
                     f'<text x="{left+plot_w-4}" y="{max(70,y_threshold-7):.2f}" text-anchor="end" font-size="13" font-family="Arial" fill="#a16b00">75% threshold</text>'])
    bits.extend([f'<text x="{left+plot_w/2}" y="345" text-anchor="middle" font-size="15" font-family="Arial" fill="#526173">Test image index (1–{len(vals)})</text>', '</svg>'])
    return svg_data('<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="380" viewBox="0 0 1280 380">'+''.join(bits))


def add_interactive_material_views(report_path, rows):
    # Enhance saved CSV report without plotting libraries.
    source=report_path.read_text(encoding='utf-8')
    import re
    # Match a complete single result-table row, preserving existing thumbnails and metadata.
    row_tags=list(re.finditer(r'<tr data-status="([^"]+)">', source))
    if len(row_tags)!=len(rows):
        raise ValueError(f'Expected {len(rows)} table rows, found {len(row_tags)}; report template changed')
    for match,row in reversed(list(zip(row_tags, rows))):
        material=html.escape(row['expected_material'].lower(), quote=True)
        source=source[:match.start()]+match.group(0)[:-1]+f' data-material="{material}">'+source[match.end():]

    # SVG chart is produced at report-generation time in the legacy template.
    # Replace only the outcome-distribution image; retain latency/confidence plots.
    chart_images=list(re.finditer(r'<section><img class="chart" src="[^"]+" alt="([^"]+) chart"></section>',source))
    pies=[m for m in chart_images if m.group(1)=='pie']
    if len(pies)!=1:
        raise ValueError('Cannot locate unique outcome distribution section in report')
    pie_html='''<section id="outcome-section" aria-label="Outcome distribution by material">
      <h2>Test outcome distribution</h2>
      <p class="muted">Choose a material to update this distribution and the image-results table together.</p>
      <div id="material-controls" class="material-controls" role="group" aria-label="Filter material">
        <button type="button" data-material-choice="all" class="selected">All materials</button>
        <button type="button" data-material-choice="cardboard">Cardboard</button>
        <button type="button" data-material-choice="glass">Glass</button>
        <button type="button" data-material-choice="metal">Metal</button>
        <button type="button" data-material-choice="paper">Paper</button>
        <button type="button" data-material-choice="plastic">Plastic</button>
        <button type="button" data-material-choice="trash">Trash</button>
      </div>
      <div class="distribution-layout">
        <div class="outcome-ring" id="outcome-ring" role="img" aria-label="Outcome counts"><div class="ring-label"><strong id="ring-total">100</strong><span>images</span></div></div>
        <div><h3 id="outcome-heading">All materials</h3><div id="outcome-legend" aria-live="polite"></div></div>
      </div>
      <h3>Breakdown by six material classes</h3>
      <p class="muted">Select a row below to view only those image tests. Each colored segment represents its share of that material's images.</p>
      <div id="material-breakdown" class="material-breakdown"></div>
    </section>'''
    source=source[:pies[0].start()]+pie_html+source[pies[0].end():]
    source=source.replace('<h2>All '+str(len(rows))+' image tests</h2>', '<h2 id="test-table-heading">All '+str(len(rows))+' image tests</h2>')
    style='''<style id="material-filter-styles">
    .material-controls{display:flex;gap:7px;flex-wrap:wrap;margin:13px 0 21px}
    .material-controls button,.material-row{font:inherit;cursor:pointer;border:1px solid #bfcbd9;background:#fff;color:#172437;border-radius:9px;padding:8px 11px}
    .material-controls button.selected,.material-row.selected{background:#173d67;color:#fff;border-color:#173d67}
    .material-controls button:focus-visible,.material-row:focus-visible{outline:3px solid #60a5fa;outline-offset:2px}
    .distribution-layout{display:grid;grid-template-columns:minmax(170px,250px) minmax(210px,1fr);gap:22px;align-items:center}
    .outcome-ring{width:min(100%,230px);aspect-ratio:1;border-radius:50%;display:grid;place-items:center;background:#e5e7eb;margin:10px auto}
    .ring-label{border-radius:50%;width:67%;height:67%;background:white;display:flex;flex-direction:column;align-items:center;justify-content:center}
    .ring-label strong{font-size:30px}.ring-label span{color:#526173}
    .outcome-legend-row{display:flex;align-items:center;gap:9px;margin:10px 0}.outcome-legend-row strong{margin-left:auto}
    .color-dot{display:inline-block;width:13px;height:13px;border-radius:4px;flex:none}
    .material-breakdown{display:grid;gap:7px}.material-row{display:grid;grid-template-columns:100px 50px minmax(100px,1fr);gap:12px;align-items:center;text-align:left;width:100%}
    .material-row.selected .bar-track{background:#315a85}.bar-track{display:flex;height:17px;overflow:hidden;border-radius:5px;background:#e4e9f0}
    .bar-segment{height:100%}#visible-count{color:#526173}
    @media(max-width:650px){.distribution-layout{grid-template-columns:1fr}.material-row{grid-template-columns:75px 35px minmax(70px,1fr)}}
    </style>'''
    source=source.replace('</head>',style+'</head>',1)
    # Existing inline JS remains the only author of table visibility; replace it
    # with a consolidated handler including material, status, and text filters.
    start=source.find('<script>const filter=document.querySelector(\'#filter\')')
    end=source.find('</script>', start)
    if start<0 or end<0:
        raise ValueError('Could not find expected original table filter script')
    js=r'''<script>
    (() => {
      const colors = {'Passed (Fully)':'#16833a','Passed (w/ Concern)':'#2563eb','Failed':'#dc2626','Ignored':'#eab308'};
      const statuses = Object.keys(colors);
      const materials = ['cardboard','glass','metal','paper','plastic','trash'];
      const rows = [...document.querySelectorAll('tbody tr[data-material]')];
      const statusSelect = document.getElementById('filter');
      const query = document.getElementById('q');
      const buttons = [...document.querySelectorAll('[data-material-choice]')];
      let selected = 'all';
      const countsFor = subset => Object.fromEntries(statuses.map(s => [s, subset.filter(r => r.dataset.status === s).length]));
      const counts = Object.fromEntries(materials.map(m => [m,countsFor(rows.filter(r=>r.dataset.material===m))]));
      function draw() {
        const subset = selected==='all' ? rows : rows.filter(r=>r.dataset.material===selected);
        const count = countsFor(subset);
        const total = subset.length;
        document.getElementById('ring-total').textContent=total;
        document.getElementById('outcome-heading').textContent=selected==='all'?'All materials':selected.charAt(0).toUpperCase()+selected.slice(1);
        let cursor=0;
        const wedges=[];
        statuses.forEach(s=>{const pc=total ? count[s]/total*100 : 0;if(pc){wedges.push(`${colors[s]} ${cursor}% ${cursor+pc}%`);cursor+=pc;}});
        document.getElementById('outcome-ring').style.background=wedges.length?`conic-gradient(${wedges.join(',')})`:'#e5e7eb';
        document.getElementById('outcome-ring').setAttribute('aria-label',statuses.map(s=>`${s}: ${count[s]}`).join(', '));
        const legend=document.getElementById('outcome-legend');legend.replaceChildren();
        statuses.forEach(s=>{const d=document.createElement('div');d.className='outcome-legend-row';
          const dot=document.createElement('span');dot.className='color-dot';dot.style.backgroundColor=colors[s];
          const label=document.createElement('span');label.textContent=s;
          const n=document.createElement('strong');n.textContent=`${count[s]} (${total?(100*count[s]/total).toFixed(1):'0.0'}%)`;
          d.append(dot,label,n);legend.appendChild(d);});
        buttons.forEach(b=>{const isSelected=b.dataset.materialChoice===selected;b.classList.toggle('selected',isSelected);b.setAttribute('aria-pressed',String(isSelected));});
        const breakdown=document.getElementById('material-breakdown');breakdown.replaceChildren();
        materials.forEach(m=>{const c=counts[m];const n=statuses.reduce((sum,s)=>sum+c[s],0);
          const button=document.createElement('button');button.type='button';button.className='material-row'+(selected===m?' selected':'');button.setAttribute('aria-pressed',String(selected===m));
          const name=document.createElement('strong');name.textContent=m[0].toUpperCase()+m.slice(1);
          const size=document.createElement('span');size.textContent=String(n);
          const track=document.createElement('span');track.className='bar-track';
          statuses.forEach(s=>{if(c[s]){const part=document.createElement('span');part.className='bar-segment';part.style.width=`${100*c[s]/n}%`;part.style.background=colors[s];part.title=`${s}: ${c[s]}`;track.appendChild(part);}});
          button.append(name,size,track);button.addEventListener('click',()=>{selected=m;draw();});breakdown.appendChild(button);
        });
        updateTable();
      }
      function updateTable(){let visible=0;
        rows.forEach(r=>{const match=(selected==='all'||selected===r.dataset.material)&&(statusSelect.value==='all'||statusSelect.value===r.dataset.status)&&r.textContent.toLowerCase().includes(query.value.trim().toLowerCase());r.style.display=match?'':'none';if(match)visible++;});
        document.getElementById('test-table-heading').textContent=`Image tests — ${visible} shown of ${rows.length}`;
      }
      buttons.forEach(b=>b.addEventListener('click',()=>{selected=b.dataset.materialChoice;draw();}));
      statusSelect.addEventListener('change',updateTable);query.addEventListener('input',updateTable);
      draw();
    })();
    </script>'''
    source=source[:start]+js+source[end+len('</script>'):]
    report_path.write_text(source,encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('results_100'))
    p.add_argument('--images',type=Path,default=Path('heldout_neoavara/images'))
    args=p.parse_args()
    csv_path=args.results/'results.csv'
    if not csv_path.is_file(): p.error(f'Missing {csv_path}')
    with csv_path.open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    if not rows: p.error('results.csv contains no predictions')
    required={'id','image','expected_material','expected_bin','predicted_material','predicted_bin','confidence','elapsed_ms','status','reason'}
    if required-set(rows[0]): p.error('Missing required CSV columns: '+', '.join(sorted(required-set(rows[0]))))
    for r in rows:
        if r['status'] not in COLORS: p.error('Invalid status for '+r['id'])
        r['confidence']=float(r['confidence']); r['elapsed_ms']=float(r['elapsed_ms'])
    summary_path=args.results/'summary.json'
    summary=json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.is_file() else {}
    original.chart_uri=svg_chart # No matplotlib import and no model inference
    output=args.results/'SmartSort_100_Image_Report.html'
    original.make_html(rows, output, {'images_root':str(args.images), 'created':summary.get('created','Saved CSV results'), 'device':summary.get('device','See saved benchmark'), 'checkpoint':'Existing 6-class checkpoint (saved CSV)'})
    add_interactive_material_views(output, rows)
    counts=Counter(r['status'] for r in rows)
    print('Report created:', output.resolve())
    print('Total:',len(rows),'Statuses:',dict(counts))
    print('Open with: Start-Process .\\results_100\\SmartSort_100_Image_Report.html')

if __name__=='__main__': main()
