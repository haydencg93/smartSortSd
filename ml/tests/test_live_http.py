#!/usr/bin/env python3
"""Launch the unchanged FastAPI app over REAL loopback HTTP and test the supplied checkpoint.
Not a physical camera/display test or material-accuracy benchmark.
"""
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image,ImageDraw

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def get_port():
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1',0))
        return sock.getsockname()[1]


def make_image(i):
    im=Image.new('RGB',(224,224),((i*39)%256,(i*57+100)%256,(i*17+50)%256))
    ImageDraw.Draw(im).ellipse((20+i%10,20,195,190),outline=(255,255,255),width=9)
    buffer=io.BytesIO();im.save(buffer,format='JPEG',quality=85)
    return buffer.getvalue()


def main():
    port=get_port();url=f'http://127.0.0.1:{port}'
    out=HERE/'results'/'live_http';out.mkdir(parents=True,exist_ok=True)
    logs=(out/'server.log').open('w')
    env=os.environ.copy();env['OMP_NUM_THREADS']='2';env['MKL_NUM_THREADS']='2'
    proc=subprocess.Popen([sys.executable,'-m','uvicorn','app:app','--host','127.0.0.1','--port',str(port),'--log-level','error'],
                          cwd=ROOT,env=env,stdout=logs,stderr=subprocess.STDOUT)
    try:
        ready=False
        for _ in range(90):
            if proc.poll() is not None:raise RuntimeError('Server terminated during startup')
            try:
                health=requests.get(url+'/health',timeout=.75)
                if health.ok and health.json().get('model_loaded'):
                    ready=True;break
            except requests.RequestException:pass
            time.sleep(.15)
        if not ready:raise RuntimeError('Server did not become healthy within ~15 seconds')
        records=[]
        for i in range(35):
            data=make_image(i)
            t=time.perf_counter()
            r=requests.post(url+'/classify',files={'file':('input.jpg',data,'image/jpeg')},timeout=8)
            ms=(time.perf_counter()-t)*1000
            r.raise_for_status()
            j=r.json();assert j['classification'] in {'trash','recycling'}
            records.append({'request_id':i+1,'http_round_trip_ms':round(ms,3),
                            'server_reported_ms':j['latency_ms'],'material':j['material'],
                            'confidence':j['confidence'],'status_code':r.status_code})
        error=requests.post(url+'/classify',files={'file':('x.txt',b'bad','text/plain')},timeout=8)
        assert error.status_code==400
        df=pd.DataFrame(records);df.to_csv(out/'http_predictions.csv',index=False)
        summary={'checkpoint':'smart_sort_mobilenetv2_6class.pth','endpoint':'localhost actual HTTP server',
                 'n_valid_requests':len(records),'expected_error_response_400':error.status_code,
                 'http_round_trip_ms':{k:round(float(np.percentile(df.http_round_trip_ms,v)),2) for k,v in [('p50',50),('p95',95),('p99',99)]},
                 'server_reported_latency_p95_ms':round(float(np.percentile(df.server_reported_ms,95)),2),
                 'all_under_5s':bool((df.http_round_trip_ms<5000).all()),
                 'limitations':'Local loopback: still excludes presence detection, settling, camera, Wi-Fi/ESP32 link and LCD; synthetic images, no ground-truth accuracy.'}
        (out/'http_metrics.json').write_text(json.dumps(summary,indent=2))
        x=np.arange(len(df))
        fig,ax=plt.subplots(figsize=(12,5.5))
        ax.plot(df.request_id,df.http_round_trip_ms,label='HTTP round trip',marker='.',markersize=5)
        ax.plot(df.request_id,df.server_reported_ms,label='Server reported inference',alpha=.9)
        ylim=max(80,float(df.http_round_trip_ms.max()*1.25))
        ax.set(xlabel='Request sequence',ylabel='Milliseconds',ylim=(0,ylim),title='FastAPI actual localhost HTTP benchmark (35 synthetic JPEG requests)')
        ax.text(0.99,.95,'5,000 ms system threshold lies beyond plot scale',transform=ax.transAxes,ha='right',va='top',fontsize=10,
                bbox={'facecolor':'#f3f5f7','edgecolor':'none','pad':5})
        ax.legend(loc='upper right',bbox_to_anchor=(1,.86));ax.grid(axis='y',alpha=.2)
        fig.tight_layout();fig.savefig(out/'http_latency.png',dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig)
        print(json.dumps(summary,indent=2));print('Output:',out)
    finally:
        proc.terminate()
        try:proc.wait(timeout=8)
        except subprocess.TimeoutExpired:proc.kill()
        logs.close()

if __name__=='__main__':main()
