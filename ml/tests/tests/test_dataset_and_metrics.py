from pathlib import Path
from PIL import Image
import pytest
from ss_testing.core import scan_dataset, duplicates_vs_training, CLASSES
from ss_testing.reporting import wilson


def test_realwaste_label_mapping_and_ignored_categories(tmp_path):
    base=tmp_path/'realwaste-main'/'RealWaste'
    for label in ['Cardboard','Glass','Metal','Paper','Plastic','Miscellaneous Trash','Food Organics']:
        d=base/label;d.mkdir(parents=True)
        Image.new('RGB',(8,8),color=(1,2,3)).save(d/'test.jpg')
    rows,missing=scan_dataset(tmp_path,'realwaste')
    assert len(rows)==6 and not missing
    assert {s.label for s in rows}==set(CLASSES)


def test_imagefolder_mapping(tmp_path):
    for label in CLASSES:
        d=tmp_path/label;d.mkdir()
        Image.new('RGB',(8,8)).save(d/'test.png')
    rows,missing=scan_dataset(tmp_path,'imagefolder')
    assert len(rows)==6 and missing==[]


def test_requires_valid_dataset(tmp_path):
    with pytest.raises(ValueError):scan_dataset(tmp_path,'realwaste')


def test_exact_duplicate_detection(tmp_path):
    d=tmp_path/'train'/'paper';d.mkdir(parents=True)
    e=tmp_path/'eval'/'paper';e.mkdir(parents=True)
    Image.new('RGB',(13,13),(20,30,40)).save(d/'one.png')
    Image.new('RGB',(13,13),(20,30,40)).save(e/'two.png')
    rows,missing=scan_dataset(tmp_path/'eval','imagefolder')
    assert len(rows)==1 and len(missing)==5
    from ss_testing.core import Sample
    assert duplicates_vs_training([Sample(e/'two.png','paper','paper/two.png')],tmp_path/'train')==['paper/two.png']


def test_wilson_interval_and_unbalanced_classes():
    lower,upper=wilson(75,100)
    assert lower<.75<upper
    assert wilson(0,0)==[None,None]
