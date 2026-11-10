"""Train Smart Sort on ONE centralized dataset (raw_data).

  python train_data.py                        # NEW model run on ALL raw_data
  python train_data.py "New Dataset Folder"   # IMPORT, then fine-tune on ALL data
  python train_data.py "New Dataset Folder" --license CC0 --epochs 25

No-argument training initializes a fresh MobileNetV2 classifier with standard
ImageNet pretrained features by default; it DOES NOT load the last Smart Sort
checkpoint. Use --from-scratch for random weights throughout. Incremental
training uses the previous Smart Sort checkpoint when available and ALWAYS
trains using all registered material classes, not just the new images.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models

OLD_CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
NORMALIZE = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
TRAIN_TRANSFORM = transforms.Compose([
    # Real campus clutter / handheld viewpoints: preserve substantial subject
    # area, but augment position, partial framing and natural lighting.
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop((224, 224), scale=(.72, 1.0), ratio=(.80, 1.25)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=12, translate=(.06, .06), scale=(.92, 1.06)),
    transforms.RandomApply([transforms.GaussianBlur(3, sigma=(.1, .6))], p=.12),
    transforms.ColorJitter(brightness=.22, contrast=.20, saturation=.16, hue=.035),
    transforms.ToTensor(), NORMALIZE,
])
EVAL_TRANSFORM = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), NORMALIZE])


def checkpoint_classes(checkpoint, state_dict):
    """A metadata sidecar is authoritative; legacy 6-class models use known order."""
    sidecar = checkpoint.with_suffix('.classes.json')
    if sidecar.is_file():
        parsed = json.loads(sidecar.read_text(encoding='utf-8'))
        classes = parsed['classes'] if isinstance(parsed, dict) else parsed
        if len(classes) != len(state_dict['classifier.1.weight']):
            raise ValueError(f'Class metadata does not match checkpoint: {sidecar}')
        return classes
    old_count = state_dict['classifier.1.weight'].shape[0]
    if old_count == len(OLD_CLASSES):
        return OLD_CLASSES
    raise ValueError(f'Cannot determine labels for {checkpoint} ({old_count} outputs). '
                     f'Place the matching {sidecar.name} alongside the checkpoint.')


def load_initial_weights(model, checkpoint, new_classes):
    """Preserve all feature weights and match old classifier rows by class name."""
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    if not isinstance(state, dict) or 'classifier.1.weight' not in state:
        raise ValueError(f'Not a MobileNetV2 model state_dict: {checkpoint}')
    old_classes = checkpoint_classes(checkpoint, state)
    expected_size = model.classifier[1].weight.shape[1]
    if state['classifier.1.weight'].shape[1] != expected_size:
        raise ValueError('Checkpoint classifier architecture does not match MobileNetV2')
    # The new head has a different output count and/or alphabetical ordering.
    # Keep its initial rows for new classes, transfer rows for shared classes.
    head_weight = state.pop('classifier.1.weight')
    head_bias = state.pop('classifier.1.bias')
    incompatible = model.load_state_dict(state, strict=False)
    if set(incompatible.missing_keys) != {'classifier.1.weight', 'classifier.1.bias'} or incompatible.unexpected_keys:
        raise ValueError(f'Checkpoint features incompatible: {incompatible}')
    with torch.no_grad():
        for label in set(old_classes) & set(new_classes):
            src, dst = old_classes.index(label), new_classes.index(label)
            model.classifier[1].weight[dst].copy_(head_weight[src])
            model.classifier[1].bias[dst].copy_(head_bias[src])
    print('Loaded prior features and classifier weights for shared classes:',
          sorted(set(old_classes) & set(new_classes)))
    print('Newly initialized classifier rows:', sorted(set(new_classes) - set(old_classes)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('new_dataset', nargs='?', type=Path,
                   help='If supplied, add this directory to raw_data BEFORE training')
    p.add_argument('--add', type=Path, help='Alternative to positional new_dataset')
    p.add_argument('--data', '--raw-data', dest='data', type=Path, default=Path('raw_data'))
    p.add_argument('--license', help='Declared license for incoming directory (CC0, MIT, etc.)')
    p.add_argument('--allow-non-cc0', action='store_true',
                   help='Explicit permission to import MIT/CC BY images after license review')
    p.add_argument('--source-url', default='',
                   help='Dataset URL to retain in the source provenance manifest')
    p.add_argument('--output', type=Path, default=Path('smart_sort_mobilenetv2_combined.pth'))
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--overwrite-output', action='store_true',
                   help='Opt in to overwriting an existing trained checkpoint')
    p.add_argument('--min-val-per-class', type=int, default=5,
                   help='Require at least this many validation images per class')
    p.add_argument('--early-stop-patience', type=int, default=8,
                   help='Stop when validation macro-F1 does not improve for this many epochs')
    p.add_argument('--init-checkpoint', type=Path,
                   help='Override initial weights; requires matching classes sidecar for expanded models')
    p.add_argument('--from-scratch', action='store_true',
                   help='No ImageNet weights (normally prefer --init-checkpoint instead)')
    args = p.parse_args()
    if args.new_dataset and args.add:
        p.error('Supply only one new dataset path (positional or --add)')
    incoming = args.new_dataset or args.add
    if args.output.exists() and not args.overwrite_output:
        p.error(f'Output checkpoint already exists: {args.output}. Use a different --output '
                'path, or explicitly pass --overwrite-output after backing up your model.')
    if args.from_scratch and args.init_checkpoint:
        p.error('Choose either --from-scratch or --init-checkpoint')
    if args.epochs < 1 or args.batch_size < 1 or args.min_val_per_class < 1 or args.early_stop_patience < 1:
        p.error('epochs and batch size must be positive')
    if incoming:
        # Import FIRST, so the following training pass includes ALL past images.
        # This safely MOVES the source directory under raw_data/sources.
        from setup_data import prepare
        print('INCREMENTAL MODE: importing new data, deduplicating, and rebuilding all train/val splits.', flush=True)
        prepare(raw_data=args.data, add=incoming,
                license_name=args.license, allow_non_cc0=args.allow_non_cc0,
                seed=args.seed, source_url=args.source_url)
        if not args.init_checkpoint and not args.from_scratch:
            candidates = [args.output, Path('smart_sort_mobilenetv2_6class.pth')]
            args.init_checkpoint = next((p for p in candidates if p.is_file()), None)
        print('INCREMENTAL MODE: training on ALL cumulative data, not only the incoming folder.', flush=True)
    else:
        print('FRESH MODE: retraining all centralized material classes; no previous Smart Sort model is loaded unless --init-checkpoint is specified.', flush=True)
    if not (args.data / 'train').is_dir() or not (args.data / 'val').is_dir():
        p.error(f'Missing prepared train/val folders in {args.data}. Run setup_data.py first.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train = datasets.ImageFolder(args.data / 'train', transform=TRAIN_TRANSFORM)
    val = datasets.ImageFolder(args.data / 'val', transform=EVAL_TRANSFORM)
    classes = train.classes
    if val.classes != classes:
        raise ValueError(f'Different train and validation labels: {classes} vs {val.classes}')
    # A prepared dataset may predate the current item-level class taxonomy.
    # Refuse to train a newer model silently with the old generic material labels.
    from setup_data import CLASSES as SUPPORTED_CLASSES
    unknown_classes = sorted(set(classes) - set(SUPPORTED_CLASSES))
    if unknown_classes:
        raise ValueError(f'Prepared dataset has obsolete/unrecognized classes: {unknown_classes}. '
                         'Run the updated setup_data.py to rebuild raw_data/train and raw_data/val.')
    if len(classes) < 2 or not len(train) or not len(val):
        raise ValueError('The combined dataset requires at least two classes with train and val samples')
    manifest_path = args.data / 'source_manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('classes') != classes:
            raise ValueError('Class folders do not match setup_data.py source_manifest.json')
        if 'not verified CC0' in manifest.get('license_policy', ''):
            print('WARNING: Combined dataset includes MIT/unknown-license images; not CC0-only.')
    counts = np.bincount(train.targets, minlength=len(classes))
    val_counts = np.bincount(val.targets, minlength=len(classes))
    if (counts == 0).any() or (val_counts < args.min_val_per_class).any():
        raise ValueError('Every included class must have training and at least '
                         f'{args.min_val_per_class} validation photographs')
    print(f'Training {len(classes)} classes: {classes}')
    print(f'Train: {len(train)} images; val: {len(val)} images')
    for label, ct_train, ct_val in zip(classes, counts, val_counts):
        print(f'  {label}: train={ct_train}, val={ct_val}')
        if ct_train < 20 or ct_val < 5:
            print(f'  WARNING: {label} has very few independent examples; accuracy may be unreliable.')

    # Tempered rather than fully inverse-frequency sampling: a class with a
    # handful of photos must not be repeated hundreds of times each epoch.
    inverse_root = np.sqrt(counts.mean() / counts)
    inverse_root = np.clip(inverse_root, 0.5, 4.0)
    sample_weights = [float(inverse_root[y]) for y in train.targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train),
                                    replacement=True)
    train_loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler,
                              num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=torch.cuda.is_available())
    if args.init_checkpoint or args.from_scratch:
        model = models.mobilenet_v2(weights=None)
    else:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    if args.init_checkpoint:
        print('Starting from checkpoint:', args.init_checkpoint)
        load_initial_weights(model, args.init_checkpoint, classes)
    else:
        print('Starting a fresh Smart Sort model:',
              'random weights' if args.from_scratch else 'ImageNet-pretrained backbone')
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.features[-8:].parameters():
        param.requires_grad = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    model = model.to(device)
    optimizer = torch.optim.AdamW([
        {'params': model.features[-8:].parameters(), 'lr': args.lr * .25},
        {'params': model.classifier.parameters(), 'lr': args.lr}
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()
    best_f1 = -1.0
    best_epoch = None
    patience = 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)
        model.eval()
        cm = torch.zeros(len(classes), len(classes), dtype=torch.long)
        low_confidence_count = 0
        with torch.inference_mode():
            for images, labels in val_loader:
                logits = model(images.to(device))
                probabilities = logits.softmax(dim=1)
                low_confidence_count += int((probabilities.max(dim=1).values < .50).sum())
                predictions = logits.argmax(dim=1).cpu()
                for truth, prediction in zip(labels, predictions):
                    cm[int(truth), int(prediction)] += 1
        tp = cm.diag().float()
        precision = tp / cm.sum(0).clamp(min=1)
        recall = tp / cm.sum(1).clamp(min=1)
        f1 = (2 * precision * recall / (precision + recall).clamp(min=1e-8)).mean().item()
        accuracy = tp.sum().item() / cm.sum().item()
        low_confidence_rate = low_confidence_count / len(val)
        history.append({'epoch': epoch + 1, 'train_loss': running_loss / len(train),
                        'val_accuracy': accuracy, 'val_macro_f1': f1,
                        'val_precision_by_class': dict(zip(classes, precision.tolist())),
                        'val_recall_by_class': dict(zip(classes, recall.tolist())),
                        'val_confusion_matrix': cm.tolist(),
                        'val_counts': dict(zip(classes, val_counts.astype(int).tolist())),
                        'val_below_50pct_confidence_rate': low_confidence_rate,
                        'engineering_85pct_accuracy_goal_met': accuracy >= .85,
                        'engineering_max_5pct_below_50pct_confidence_goal_met':
                            low_confidence_rate <= .05})
        print(f'Epoch {epoch+1}/{args.epochs}: val accuracy={accuracy:.2%} '
              f'macro-F1={f1:.3f} below-50%-confidence={low_confidence_rate:.2%}',
              flush=True)
        if f1 > best_f1 + 1e-5:
            best_f1 = f1
            best_epoch = epoch + 1
            patience = 0
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output)
            args.output.with_suffix('.classes.json').write_text(json.dumps({
                'classes': classes,
                'dataset_manifest': str(manifest_path.resolve()) if manifest_path.exists() else None,
                'init_checkpoint': str(args.init_checkpoint) if args.init_checkpoint else None,
            }, indent=2), encoding='utf-8')
        else:
            patience += 1
        scheduler.step()
        if patience >= args.early_stop_patience:
            print(f'Early stopping: no macro-F1 improvement for {patience} epochs.', flush=True)
            break
    args.output.with_suffix('.metrics.json').write_text(json.dumps({
        'classes': classes, 'best_val_macro_f1': best_f1,
        'best_epoch': best_epoch, 'history': history,
        'init_checkpoint': str(args.init_checkpoint) if args.init_checkpoint else None,
        'pretraining': ('prior_checkpoint' if args.init_checkpoint else 'none' if args.from_scratch
                        else 'torchvision_imagenet'),
    }, indent=2), encoding='utf-8')
    print('Saved best checkpoint:', args.output)
    print('Class order saved:', args.output.with_suffix('.classes.json'))


if __name__ == '__main__':
    main()
