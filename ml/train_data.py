"""Fine-tune six-class MobileNetV2 using the output of setup_data.py.

Preserves the six label order and API architecture. Saves the BEST validation
checkpoint rather than always saving the last epoch. No benchmark images used.
Example: python train_data.py --data data_6class --epochs 25
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

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
NORMALIZE = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.GaussianBlur(3, sigma=(.1, .8))], p=.15),
    transforms.ColorJitter(brightness=.25, contrast=.25, saturation=.15, hue=.04),
    transforms.ToTensor(), NORMALIZE,
])
EVAL_TRANSFORM = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), NORMALIZE])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('data_6class'))
    p.add_argument('--output', type=Path, default=Path('smart_sort_mobilenetv2_6class_improved.pth'))
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--init-checkpoint', type=Path, help='Optional existing six-class checkpoint; retains its training provenance')
    p.add_argument('--from-scratch', action='store_true', help='No pretrained weights; useful if pretrained data provenance is restricted')
    args = p.parse_args()
    if args.from_scratch and args.init_checkpoint:
        p.error('Choose either --from-scratch or --init-checkpoint')
    if args.epochs < 1 or args.batch_size < 1:
        p.error('epochs and batch size must be positive')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    train = datasets.ImageFolder(args.data / 'train', transform=TRAIN_TRANSFORM)
    val = datasets.ImageFolder(args.data / 'val', transform=EVAL_TRANSFORM)
    if train.classes != CLASSES or val.classes != CLASSES:
        raise ValueError(f'Train/val must contain exactly these six classes: {CLASSES}')
    if not train or not val:
        raise ValueError('Empty train/val dataset')
    counts = np.bincount(train.targets, minlength=6)
    if (counts == 0).any():
        raise ValueError('Each material class needs training images')
    sample_weights = [1 / counts[y] for y in train.targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train), replacement=True)
    train_loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    if args.init_checkpoint or args.from_scratch:
        model = models.mobilenet_v2(weights=None)
    else:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
    if args.init_checkpoint:
        model.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu', weights_only=True), strict=True)
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.features[-8:].parameters():
        param.requires_grad = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.AdamW([{'params': model.features[-8:].parameters(), 'lr': args.lr * .25},
                                   {'params': model.classifier.parameters(), 'lr': args.lr}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()
    best_f1 = -1.; history = []
    for epoch in range(args.epochs):
        model.train()
        # Prevent a tiny new dataset from rewriting all running BatchNorm stats.
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        running_loss = 0.
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward(); optimizer.step()
            running_loss += loss.item() * len(labels)
        model.eval(); cm = torch.zeros(6, 6, dtype=torch.long); correct = total = 0
        with torch.inference_mode():
            for images, labels in val_loader:
                logits = model(images.to(device))
                pred = logits.argmax(dim=1).cpu()
                for truth, prediction in zip(labels, pred):
                    cm[int(truth), int(prediction)] += 1
                correct += (pred == labels).sum().item(); total += len(labels)
        tp = cm.diag().float()
        precision = tp / cm.sum(0).clamp(min=1)
        recall = tp / cm.sum(1).clamp(min=1)
        f1 = (2 * precision * recall / (precision + recall).clamp(min=1e-8)).mean().item()
        accuracy = correct / total
        history.append({'epoch': epoch + 1, 'train_loss': running_loss / len(train),
                        'val_accuracy': accuracy, 'val_macro_f1': f1,
                        'val_recall_by_class': dict(zip(CLASSES, recall.tolist()))})
        print(f'Epoch {epoch+1}/{args.epochs}: accuracy={accuracy:.2%} macro-F1={f1:.3f}')
        if f1 > best_f1:
            best_f1 = f1
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output)
        scheduler.step()
    args.output.with_suffix('.metrics.json').write_text(json.dumps({'classes': CLASSES,
        'best_val_macro_f1': best_f1, 'history': history,
        'init_checkpoint': str(args.init_checkpoint) if args.init_checkpoint else None,
        'pretraining': 'none' if args.from_scratch else ('prior_checkpoint' if args.init_checkpoint else 'torchvision_imagenet')}, indent=2))
    print('Saved best checkpoint:', args.output)


if __name__ == '__main__':
    main()
