"""Create a SHA256 inventory of known training/validation image roots."""
import argparse,hashlib
from pathlib import Path
SUFFIXES={'.jpg','.jpeg','.png','.bmp','.webp'}
p=argparse.ArgumentParser();p.add_argument('roots',nargs='+',type=Path);p.add_argument('--output',type=Path,default=Path('training_hashes.txt'));args=p.parse_args()
hashes=set()
for root in args.roots:
    if not root.exists(): raise SystemExit(f'Missing data root: {root}')
    for img in root.rglob('*'):
        if img.is_file() and img.suffix.lower() in SUFFIXES:
            h=hashlib.sha256(img.read_bytes()).hexdigest();hashes.add(h)
args.output.write_text('\n'.join(sorted(hashes))+'\n')
print(f'Wrote {len(hashes)} exact training-image hashes to {args.output}')
