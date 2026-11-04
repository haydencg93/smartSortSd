"""Populate a pre-existing NeoAivara manifest with PROVISIONAL material-rule bin labels.
No download required. Uses a .bak backup and never alters the source photos.
"""
import argparse, csv, shutil
from pathlib import Path
from prepare_neoavara import AUTO_BIN, CITY_RULE_URL

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--overwrite-reviewed',action='store_true',help='DANGEROUS: replace manually verified labels too')
    a=p.parse_args()
    if not a.manifest.is_file(): p.error(f'Missing manifest: {a.manifest}')
    with a.manifest.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f); fields=reader.fieldnames; rows=list(reader)
    if not fields or not all(x in fields for x in ('expected_material','expected_bin','reviewed','reviewer','rule_source','notes')):
        p.error('Manifest missing required columns')
    changes=0
    for row in rows:
        material=row['expected_material'].strip().lower()
        if material not in AUTO_BIN: p.error(f'Unknown material: {material}')
        if row.get('reviewed','').strip().lower() in ('yes','true','1') and not a.overwrite_reviewed:
            continue
        row.update(expected_bin=AUTO_BIN[material],reviewed='auto',reviewer='automated-material-policy',rule_source=CITY_RULE_URL,notes='PROVISIONAL inferred from source material only. Not item-verified; plastic subtype, contamination, and photo conditions are unknown.')
        changes+=1
    backup=a.manifest.with_suffix(a.manifest.suffix+'.bak')
    if backup.exists(): p.error(f'Backup already exists: {backup}. Move it before rerunning.')
    shutil.copy2(a.manifest,backup)
    with a.manifest.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print(f'Updated {changes} entries. Backup: {backup}')
    print('WARNING: Automatically inferred bin labels are PROVISIONAL, not image-verified Iowa City disposal ground truth.')
if __name__=='__main__': main()
