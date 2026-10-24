#!/usr/bin/env python3
"""Exercise the genuine uploaded model via FastAPI using synthetic diagnostic patterns.
These are NOT photographs of waste; no real-world accuracy claims are produced.
"""
import importlib.util
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image,ImageDraw,ImageFilter
from fastapi.testclient import TestClient

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE))
from ss_testing.core import CLASSES,RULES
from ss_testing.reporting import savefig


def examples():
    examples=[]
    c={'white':(255,255,255),'black':(0,0,0),'red':(230,40,40),'blue':(35,85,215),'green':(35,190,100),
       'brown':(145,90,35),'gray':(125,125,125),'yellow':(230,210,45)}
    for name,rgb in c.items():
        examples.append((f'solid {name}',Image.new('RGB',(256,256),rgb)))
    for name,kind in [('circle on white','circle'),('rectangle on white','rectangle'),('stripes','stripes'),('noise','noise')]:
        im=Image.new('RGB',(256,256),'white');draw=ImageDraw.Draw(im)
        if kind=='circle':draw.ellipse((45,45,210,210),fill='gray')
        if kind=='rectangle':draw.rectangle((55,55,205,205),fill='brown')
        if kind=='stripes':
            for i in range(0,256,20):draw.rectangle((i,0,i+8,255),fill='black')
        if kind=='noise':
            rng=np.random.default_rng(14)
            im=Image.fromarray(rng.integers(0,256,(256,256,3),dtype=np.uint8),'RGB')
        examples.append((name,im))
    return examples


def main():
    torch.set_num_threads(2)
    spec=importlib.util.spec_from_file_location('smart_sort_original_app_smoke',ROOT/'app.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MODEL_PATH=str(ROOT/'smart_sort_mobilenetv2_6class.pth')
    dest=HERE/'results'/'smoke';dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    with TestClient(module.app) as client:
        assert client.get('/health').json()['model_loaded']
        for name,im in examples():
            b=io.BytesIO();im.save(b,format='PNG')
            start=time.perf_counter()
            resp=client.post('/classify',files={'file':('synthetic.png',b.getvalue(),'image/png')})
            wall=(time.perf_counter()-start)*1000
            resp.raise_for_status()
            obj=resp.json()
            assert obj['classification']==RULES[obj['material']]
            rows.append({'pattern':name,**obj,'asgi_round_trip_ms':round(wall,2)})
        invalid=[('wrong content-type',b'not image','text/plain',400),('fake image',b'bad image','image/png',400)]
        checks={'health':True,'valid_images':len(rows),'invalid_cases':{}}
        for name,raw,mime,status in invalid:
            actual=client.post('/classify',files={'file':('input',raw,mime)}).status_code
            checks['invalid_cases'][name]=dict(observed=actual,expected=status,pass_test=actual==status)
    frame=pd.DataFrame(rows);frame.to_csv(dest/'synthetic_api_results.csv',index=False)
    checks.update({'synthetic_images_are_not_accuracy_data':True,
                   'asgi_latency_ms':{key:round(float(np.percentile(frame.asgi_round_trip_ms,p)),2) for key,p in [('p50',50),('p95',95),('p99',99)]},
                   'server_reported_latency_ms_p95':round(float(np.percentile(frame.latency_ms,95)),2),
                   'all_synthetic_requests_under_5sec':bool((frame.asgi_round_trip_ms<5000).all()),
                   'note':'In-process ASGI TestClient timings exclude physical camera capture, network, proximity activation and LCD display; no accuracy measured.'})
    (dest/'smoke_metrics.json').write_text(json.dumps(checks,indent=2))
    fig,ax=plt.subplots(figsize=(11,5.5))
    ax.bar(frame.pattern,frame.asgi_round_trip_ms,label='ASGI elapsed')
    ax.plot(frame.pattern,frame.latency_ms,'o',label='API reported')
    ax.set(xlabel='Synthetic diagnostic pattern',ylabel='Milliseconds',
           title='Smart Sort API timing — synthetic diagnostics, not image accuracy')
    ax.text(.99,.98,'5,000 ms system threshold lies beyond plot scale',transform=ax.transAxes,
            ha='right',va='top',fontsize=10,
            bbox={'facecolor':'#f3f5f7','edgecolor':'none','pad':5})
    ax.set_ylim(0,max(90,float(frame.asgi_round_trip_ms.max()*1.6)))
    ax.tick_params(axis='x',rotation=45);ax.legend();ax.grid(axis='y',alpha=.2)
    savefig(dest/'synthetic_latency.png')
    fig,ax=plt.subplots(figsize=(11,5.5))
    bars=ax.bar(frame.pattern,frame.confidence*100)
    ax.set(xlabel='Synthetic diagnostic pattern',ylabel='Top-material confidence (%)',ylim=(0,105),title='Model confidence on out-of-distribution diagnostic patterns — NOT accuracy')
    ax.tick_params(axis='x',rotation=45)
    for rect,label in zip(bars,frame.material):
        ax.text(rect.get_x()+rect.get_width()/2,rect.get_height()+1,label,ha='center',va='bottom',fontsize=8,rotation=40)
    savefig(dest/'synthetic_confidence_not_accuracy.png')
    # Diagnostic gallery with every expected label unknown.
    fig,axes=plt.subplots(3,4,figsize=(12,10))
    for ax,(name,im),r in zip(axes.flat,examples(),rows):
        ax.imshow(im);ax.set_title(f'{name}\nPredict: {r["material"]} ({r["confidence"]:.0%})',fontsize=9);ax.axis('off')
    fig.suptitle('Diagnostic images; NONE are labeled waste samples',fontsize=14)
    savefig(dest/'synthetic_diagnostic_gallery.png')
    print(json.dumps(checks,indent=2))
    print('Charts and CSV in',dest)

if __name__=='__main__':main()
