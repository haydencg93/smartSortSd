import importlib.util
from pathlib import Path
import sys
import pytest
import torch
from fastapi.testclient import TestClient

ROOT=Path(__file__).resolve().parents[2]
PROJECT=ROOT
MODEL_PATH=ROOT/'smart_sort_mobilenetv2_6class.pth'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

@pytest.fixture(scope='session')
def appmod():
    spec=importlib.util.spec_from_file_location('smart_sort_original_app',PROJECT/'app.py')
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    module.MODEL_PATH=str(MODEL_PATH)
    return module

@pytest.fixture(scope='session')
def client(appmod):
    torch.set_num_threads(2)
    with TestClient(appmod.app) as c:
        yield c
