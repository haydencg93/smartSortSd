"""Exercise actual pc_proto_ml.py helper functions without starting a webcam loop."""
import ast
from pathlib import Path
import numpy as np
import cv2

SOURCE=Path(__file__).resolve().parents[2]/'pc_proto_ml.py'


def load_helpers():
    tree=ast.parse(SOURCE.read_text())
    names={'to_gray_blur','presence_fraction_from'}
    funcs=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    consts={n.targets[0].id: ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {'PRESENCE_THRESHOLD','PRESENCE_NOISE_KERNEL','PRESENCE_ENTER_FRACTION','PRESENCE_EXIT_FRACTION','STILLNESS_THRESHOLD','STILLNESS_FRAMES_REQUIRED'}}
    env={'cv2':cv2,'np':np,**consts}
    env['_noise_kernel']=np.ones((env['PRESENCE_NOISE_KERNEL'],env['PRESENCE_NOISE_KERNEL']),np.uint8)
    exec(compile(ast.Module(body=funcs,type_ignores=[]),str(SOURCE),'exec'),env)
    return env


def test_blur_grayscale_shape():
    env=load_helpers();frame=np.zeros((80,80,3),dtype=np.uint8)
    assert env['to_gray_blur'](frame).shape==(80,80)


def test_empty_frame_no_presence():
    env=load_helpers();base=np.zeros((100,100),np.uint8)
    assert env['presence_fraction_from'](base,base)==0


def test_large_present_object_detected():
    env=load_helpers();base=np.zeros((100,100),np.uint8)
    present=base.copy();present[20:80,20:80]=230
    assert env['presence_fraction_from'](present,base)>env['PRESENCE_ENTER_FRACTION']


def test_isolated_pixels_removed_by_morphology():
    env=load_helpers();base=np.zeros((100,100),np.uint8)
    noisy=base.copy();noisy[::10,::10]=220
    assert env['presence_fraction_from'](noisy,base)<0.01


def test_presence_hysteresis_parameters():
    env=load_helpers()
    assert 0<env['PRESENCE_EXIT_FRACTION']<env['PRESENCE_ENTER_FRACTION']<1
    assert env['STILLNESS_FRAMES_REQUIRED']>1
