"""Unit and API-integration tests that work without original image data."""
from pathlib import Path
import io
import math
import time
import concurrent.futures

import numpy as np
import pytest
import torch
from PIL import Image,ImageDraw
from torchvision import models

from ss_testing.core import CLASSES,RULES,PREPROCESS,load_checkpoint,predict
from conftest import MODEL_PATH,ROOT


def png_bytes(mode='RGB'):
    color={'RGB':(180,80,40),'RGBA':(180,80,40,128),'L':110}[mode]
    img=Image.new(mode,(113,89),color)
    buf=io.BytesIO();img.save(buf,format='PNG')
    return buf.getvalue()


def test_checkpoint_is_valid_six_class_state_dict():
    state=torch.load(str(MODEL_PATH),map_location='cpu',weights_only=True)
    assert tuple(state['classifier.1.weight'].shape)==(6,1280)
    assert state['classifier.1.bias'].shape==(6,)


def test_other_checkpoint_is_not_six_class():
    state=torch.load(str(ROOT/'smart_sort_mobilenetv2.pth'),map_location='cpu',weights_only=True)
    assert state['classifier.1.weight'].shape[0]==2
    with pytest.raises(RuntimeError):
        load_checkpoint(ROOT/'smart_sort_mobilenetv2.pth')


def test_labels_match_imagefolder_alphabetical_order():
    assert list(CLASSES)==sorted(CLASSES)
    assert len(RULES)==6 and set(RULES)==set(CLASSES)


def test_preprocessing_is_rgb_224_and_finite():
    t=PREPROCESS(Image.new('RGB',(67,99),(100,125,150)))
    assert t.shape==(3,224,224)
    assert torch.isfinite(t).all()


def test_local_model_deterministic_and_probabilities():
    m=load_checkpoint(MODEL_PATH)
    img=Image.new('RGB',(160,120),(140,110,90))
    a=predict(m,img);b=predict(m,img)
    assert a['material']==b['material']
    assert a['probabilities']==pytest.approx(b['probabilities'],abs=1e-6)
    assert abs(sum(a['probabilities'])-1)<1e-5
    assert a['classification']==RULES[a['material']]
    assert 0<=a['confidence']<=1


def test_health_loaded(client):
    resp=client.get('/health')
    assert resp.status_code==200
    assert resp.json()=={'status':'ok','model_loaded':True}


@pytest.mark.parametrize('mode',['RGB','RGBA','L'])
def test_png_end_to_end_api(client, mode):
    resp=client.post('/classify',files={'file':('synthetic.png',png_bytes(mode),'image/png')})
    assert resp.status_code==200,resp.text
    obj=resp.json()
    assert set(obj)=={'material','confidence','classification','latency_ms'}
    assert obj['material'] in CLASSES
    assert obj['classification']==RULES[obj['material']]
    assert 0<=obj['confidence']<=1
    assert math.isfinite(obj['latency_ms']) and obj['latency_ms']>=0


def test_jpeg_end_to_end_api(client):
    img=Image.new('RGB',(600,330),(25,200,100));buf=io.BytesIO();img.save(buf,format='JPEG')
    resp=client.post('/classify',files={'file':('x.jpg',buf.getvalue(),'image/jpeg')})
    assert resp.status_code==200


def test_rejects_non_image_content_type(client):
    r=client.post('/classify',files={'file':('notimage.txt',b'plain text','text/plain')})
    assert r.status_code==400


def test_rejects_corrupt_image_even_with_image_mime(client):
    r=client.post('/classify',files={'file':('fake.png',b'not a valid png','image/png')})
    assert r.status_code==400


def test_rejects_missing_file(client):
    assert client.post('/classify').status_code==422


def test_unloaded_model_returns_503(client,appmod,monkeypatch):
    # TestClient lifespan has loaded model; temporarily toggle exactly for this call.
    monkeypatch.setattr(appmod,'model',None)
    r=client.post('/classify',files={'file':('x.png',png_bytes(),'image/png')})
    assert r.status_code==503


def test_current_iowa_city_mapping_exact(appmod):
    assert appmod.IOWA_CITY_RULES==RULES
    assert tuple(appmod.MATERIAL_CLASSES)==CLASSES
    assert appmod.IOWA_CITY_RULES['glass']=='trash'


def test_inference_request_under_five_seconds(client):
    started=time.perf_counter()
    r=client.post('/classify',files={'file':('x.png',png_bytes(),'image/png')})
    elapsed=(time.perf_counter()-started)
    assert r.status_code==200
    assert elapsed<5.0, f'API request took {elapsed:.2f}s'


def test_model_preprocess_matches_api(appmod):
    img=Image.new('RGB',(160,90),(17,133,201))
    assert torch.equal(PREPROCESS(img),appmod.preprocess(img))


def test_parallel_smoke_requests(client):
    # In-process TestClient; concurrency smoke only, not a capacity/load certificate.
    def call(_):
        r=client.post('/classify',files={'file':('x.png',png_bytes(),'image/png')})
        return r.status_code,r.json().get('material')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(call,range(6)))
    assert all(s==200 and m in CLASSES for s,m in results)
