#!/usr/bin/env python3
"""Verify the existing Kaggle Smart Sort model without automatically deploying it.

Only structural/provenance failures exit nonzero. A valid candidate below any
quality gate exits zero with approved=false. These training-set validation gates
are NOT a substitute for a truly independent, reviewed camera-image benchmark.
"""
import argparse
import ast
import hashlib
import json
import math
import os
import re
from pathlib import Path

BASENAME = 'smart_sort_mobilenetv2_combined'
VALID_DISPOSALS = {'recycling', 'landfill', 'special_disposal', 'manual_review'}


def append_github_file(env_name, lines):
    path = os.environ.get(env_name)
    if path:
        with open(path, 'a', encoding='utf-8') as stream:
            stream.write('\n'.join(lines) + '\n')


def sha256_file(path):
    # Stream large checkpoints rather than loading them twice into RAM.
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def supported_classes(setup_script):
    """Read the only source-of-truth taxonomy without running dataset preparation."""
    tree = ast.parse(setup_script.read_text(encoding='utf-8'), filename=str(setup_script))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CLASSES'
                                                 for t in node.targets):
            labels = ast.literal_eval(node.value)
            if (not isinstance(labels, (tuple, list)) or len(labels) < 2
                    or any(not isinstance(s, str) or not s for s in labels)
                    or len(set(labels)) != len(labels)):
                break
            return set(labels)
    raise SystemExit(f'Could not obtain valid CLASSES from {setup_script}')


def number(value, description):
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise SystemExit(f'Missing or malformed {description}: {value!r}') from exc
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise SystemExit(f'{description} must be finite and between 0 and 1: {result}')
    return result


def validate(args):
    import torch
    from torchvision import models

    directory = Path(args.directory)
    paths = {
        'model': directory / f'{BASENAME}.pth',
        'classes': directory / f'{BASENAME}.classes.json',
        'metrics': directory / f'{BASENAME}.metrics.json',
        'provenance': directory / 'cloud_run.json',
    }
    for name, path in paths.items():
        if not path.is_file():
            raise SystemExit(f'Missing required Kaggle output {name}: {path}')

    root = Path(__file__).resolve().parents[1]
    train_script, setup_script = root / 'train_data.py', root / 'setup_data.py'
    rules_path = root / 'rules' / 'uiowa.json'
    for path in (train_script, setup_script, rules_path):
        if not path.is_file():
            raise SystemExit(f'Missing current Smart Sort source or rules: {path}')

    class_meta = json.loads(paths['classes'].read_text(encoding='utf-8'))
    labels = class_meta.get('classes') if isinstance(class_meta, dict) else class_meta
    metrics = json.loads(paths['metrics'].read_text(encoding='utf-8'))
    run = json.loads(paths['provenance'].read_text(encoding='utf-8'))
    if (not isinstance(labels, list) or len(labels) < 2
            or not all(isinstance(s, str) and s for s in labels)
            or labels != sorted(set(labels))
            or metrics.get('classes') != labels):
        raise SystemExit('Invalid/unordered class metadata or metrics class mismatch')
    unknown = sorted(set(labels) - supported_classes(setup_script))
    if unknown:
        raise SystemExit(f'Outdated or unrecognized model class labels: {unknown}. '
                         'Rebuild the prepared dataset using the current setup_data.py.')
    rules = json.loads(rules_path.read_text(encoding='utf-8'))
    uiowa = rules.get('locations', {}).get('uiowa', {}).get('rules', {})
    missing = sorted(set(labels) - set(uiowa))
    if missing or any(uiowa[label].get('disposal') not in VALID_DISPOSALS for label in labels):
        raise SystemExit(f'No valid UIowa disposal routing for model labels: {missing}')

    digest = sha256_file(paths['model'])
    if run.get('model_sha256') != digest:
        raise SystemExit('Model SHA256 disagrees with Kaggle provenance file')
    source_commit = str(run.get('source_commit', ''))
    if not re.fullmatch(r'[0-9a-f]{40}', source_commit) or source_commit == '0' * 40:
        raise SystemExit('Run source_commit is invalid (requires an actual GitHub commit SHA)')
    for name, script in [('training_source_sha256', train_script),
                         ('setup_source_sha256', setup_script)]:
        if run.get(name) != sha256_file(script):
            raise SystemExit(f'Kaggle {name} differs from the current GitHub code. '
                             'This is a stale model candidate; launch training again.')

    history = metrics.get('history')
    if not isinstance(history, list) or not history:
        raise SystemExit('No validation history in metrics.json')
    epoch = metrics.get('best_epoch')
    try:
        if not isinstance(epoch, int) or epoch < 1:
            raise ValueError('Invalid best_epoch')
        best = next(item for item in history if item['epoch'] == epoch)
        f1 = number(best['val_macro_f1'], 'best val_macro_f1')
        acc = number(best['val_accuracy'], 'best val_accuracy')
        last_best = float(metrics['best_val_macro_f1'])
        if not math.isfinite(last_best) or abs(last_best - f1) > 2e-5:
            raise ValueError('best_val_macro_f1 disagrees with the selected best epoch')
        recall = best['val_recall_by_class']
        if not isinstance(recall, dict) or set(recall) != set(labels):
            raise ValueError('Missing class-level recall scores')
        recalls = {label: number(recall[label], f'{label} recall') for label in labels}
        counts = best['val_counts']
        if (not isinstance(counts, dict) or set(counts) != set(labels)
                or any(not isinstance(counts[label], int) or counts[label] < 5 for label in labels)):
            raise ValueError('At least five validation images per class are required')
        low_confidence = number(best['val_below_50pct_confidence_rate'],
                                'fraction of validation images with confidence below 50%')
        # Confirm best_epoch really is the best recorded macro-F1, subject to the
        # trainer's 1e-5 improvement tolerance.
        if any(number(item['val_macro_f1'], 'historical macro-F1') > f1 + 2e-5
               for item in history):
            raise ValueError('best_epoch is inconsistent with validation history')
    except (KeyError, TypeError, ValueError, StopIteration) as exc:
        raise SystemExit(f'Malformed or inconsistent validation metrics: {exc}') from exc

    # Check the model file's actual layer sizes and every tensor, not just JSON.
    state = torch.load(paths['model'], map_location='cpu', weights_only=True)
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(labels))
    model.load_state_dict(state, strict=True)

    thresholds = [args.min_accuracy, args.min_macro_f1, args.min_glass_recall,
                  args.min_trash_recall, args.min_class_recall,
                  args.max_low_confidence_rate]
    if any(not math.isfinite(t) or not 0 <= t <= 1 for t in thresholds):
        raise SystemExit('All quality thresholds must be finite and in [0, 1]')
    gates = [
        ('Accuracy', acc, args.min_accuracy, '>='),
        ('Macro-F1', f1, args.min_macro_f1, '>='),
        ('Glass recall', recalls.get('glass', 0), args.min_glass_recall, '>='),
        ('Trash recall', recalls.get('trash', 0), args.min_trash_recall, '>='),
        ('Worst-class recall', min(recalls.values()), args.min_class_recall, '>='),
        ('Below-50%-confidence image rate', low_confidence, args.max_low_confidence_rate, '<='),
    ]
    failed = [name for name, actual, minimum, direction in gates
              if (actual < minimum if direction == '>=' else actual > minimum)]
    approved = not failed
    previous = Path(args.previous)
    changed = not previous.is_file() or sha256_file(previous) != digest
    tag = 'smartsort-' + digest[:12]
    verdict = ('APPROVED for automated publication gate' if approved
               else 'REJECTED: quality gates unmet (normal review outcome)')
    print(f'Best validation epoch: {epoch}; classes={labels}; {verdict}')
    for name, actual, threshold, direction in gates:
        ok = name not in failed
        print(f'  {name}: {actual:.4f} (required {direction} {threshold:.4f}) -- '
              f'{"PASS" if ok else "FAIL"}')
    print(f'Checkpoint/source provenance verified; changed={changed}; release={tag}')
    if not approved:
        print('No publication. Valid candidate below quality thresholds.')

    append_github_file('GITHUB_OUTPUT', [
        f"approved={'true' if approved else 'false'}",
        f"changed={'true' if changed else 'false'}",
        f'tag={tag}', f'digest={digest}', f'source_commit={source_commit}',
        'failed_gates=' + ', '.join(failed),
    ])
    append_github_file('GITHUB_STEP_SUMMARY', [
        '## Smart Sort model review',
        f'**Result:** {verdict}',
        f'**Best validation epoch:** {epoch}  ',
        f'**Classes:** {len(labels)}  ',
        f'**Model digest:** `{digest}`  ',
        '', '| Quality metric | Actual | Required | Result |',
        '| --- | ---: | ---: | --- |',
        *[f'| {name} | {actual:.2%} | {direction} {threshold:.2%} | '
          f'{"PASS" if name not in failed else "REJECT"} |'
          for name, actual, threshold, direction in gates],
        '', 'Checkpoint structure, class order, source SHA256, and provenance were verified. '
        'This validation does not replace independent camera testing.',
    ])
    return approved


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--previous', default='ml/smart_sort_mobilenetv2_combined.pth')
    parser.add_argument('--min-accuracy', type=float, default=.85)
    parser.add_argument('--min-macro-f1', type=float, default=.75)
    parser.add_argument('--min-glass-recall', type=float, default=.80)
    parser.add_argument('--min-trash-recall', type=float, default=.80)
    parser.add_argument('--min-class-recall', type=float, default=.50)
    parser.add_argument('--max-low-confidence-rate', type=float, default=.05)
    validate(parser.parse_args())
