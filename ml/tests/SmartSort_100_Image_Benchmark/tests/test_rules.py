import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
from benchmark import decide,read_manifest
@pytest.mark.parametrize('m,b,pbin,expbin,c,t,status',[('paper','paper','recycling','recycling',.9,100,'Passed (Fully)'),('paper','paper','recycling','recycling',.74,100,'Passed (w/ Concern)'),('paper','paper','recycling','recycling',.9,2500.1,'Passed (w/ Concern)'),('paper','plastic','recycling','recycling',.99,1,'Failed'),('glass','glass','trash','recycling',.98,20,'Failed'),('plastic','plastic','recycling','recycling',.75,2500,'Passed (Fully)'),('trash','trash','trash','trash',.5,3200,'Passed (w/ Concern)')])
def test_outcomes(m,b,pbin,expbin,c,t,status): assert decide(m,b,pbin,expbin,c,t)[0]==status

def test_ignored_never_emitted():
    for c in (0,.75,1):
        for t in (1,2501):assert decide('metal','metal','recycling','recycling',c,t)[0]!='Ignored'

def test_manifest_missing(tmp_path):
    with pytest.raises(ValueError,match='Missing manifest'):read_manifest(tmp_path/'missing.csv',tmp_path)
