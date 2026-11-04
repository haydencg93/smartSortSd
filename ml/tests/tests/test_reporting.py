"""Tests the analytics/plotting pipeline using plainly synthetic fixture predictions.
Fixture scores are NEVER claimed as measurements of the submitted model.
"""
import json
from pathlib import Path
from PIL import Image
from ss_testing.core import CLASSES
from ss_testing.reporting import write_accuracy_report


def test_metrics_and_actual_plot_files(tmp_path):
    rows=[]
    for i,c in enumerate(CLASSES):
        p=tmp_path/f'{c}.png';Image.new('RGB',(32,32),'gray').save(p)
        pred='paper' if c=='cardboard' else c
        rows.append({'path':str(p),'relative_path':p.name,'label':c,'predicted':pred,
                     'confidence':0.8,'latency_ms':float(i+5),'actual_bin':None})
    out=tmp_path/'report'
    metrics=write_accuracy_report(rows,out,{'source':'unit-test-fixture',
                                            'complete_dataset':False,'train_overlap_status':'unknown_training_provenance'})
    assert abs(metrics['material_accuracy']-5/6)<.0001
    assert metrics['current_code_bin_accuracy']==1.0 # cardboard->paper still recycling
    assert metrics['engineering_75pct_gate']['status']=='not_certifiable'
    for name in ['material_confusion.png','material_confusion_normalized.png',
                 'routing_confusion.png','per_class_scores.png','calibration.png',
                 'latency_distribution.png','correct_gallery.png','errors_gallery.png',
                 'predictions.csv','metrics.json']:
        assert (out/name).exists(),name


def test_absent_real_policy_ground_truth_not_claimed(tmp_path):
    p=tmp_path/'x.png';Image.new('RGB',(32,32)).save(p)
    rows=[{'path':str(p),'label':'trash','predicted':'trash','confidence':1.0,
           'latency_ms':8.0,'actual_bin':None}]
    result=write_accuracy_report(rows,tmp_path/'report',{'source':'fixture','complete_dataset':False,
                                                         'train_overlap_status':'unknown_training_provenance'})
    assert 'annotated_iowa_city_bin_accuracy' not in result
