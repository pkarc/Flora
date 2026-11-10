"""Compare two segment-worker results (stdlib only): segments by one-to-one IoU matching, and raw windows if saved.

  python captioner/compare_runs.py baseline.json candidate.json [--iou 0.75]

Segments: matched pairs at IoU >= --iou, F1, and the unmatched segments with their provenance.
Raw windows (runs made with --save-raw): per window, whether the input token count and the model output are
identical; the first differences are printed. Windows are paired by (clip, scope_id).
"""
import argparse
import json
import sys


def iou(x, y):
    inter = max(0.0, min(x['end_s'], y['end_s']) - max(x['start_s'], y['start_s']))
    union = (x['end_s'] - x['start_s']) + (y['end_s'] - y['start_s']) - inter
    return inter / union if union > 0 else 0.0


def match(a, b, threshold):
    pairs = sorted(((iou(x, y), i, j) for i, x in enumerate(a) for j, y in enumerate(b)), reverse=True)
    used_a, used_b = set(), set()
    for value, i, j in pairs:
        if value < threshold:
            break
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
    return used_a, used_b


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('baseline')
    p.add_argument('candidate')
    p.add_argument('--iou', type=float, default=0.75)
    p.add_argument('--show', type=int, default=3, help='How many differing windows to print.')
    args = p.parse_args(argv)
    base, cand = (json.load(open(path)) for path in (args.baseline, args.candidate))
    a, b = base['segments'], cand['segments']
    used_a, used_b = match(a, b, args.iou)
    m = len(used_a)
    precision, recall = (m / len(b) if b else 0.0), (m / len(a) if a else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f'segments: baseline {len(a)}, candidate {len(b)}; matched at IoU>={args.iou}: {m}; F1={f1:.3f}')
    for name, rows, used in (('baseline only', a, used_a), ('candidate only', b, used_b)):
        for i, row in enumerate(rows):
            if i not in used:
                print(f'  {name}: {row["start_s"]:.2f}-{row["end_s"]:.2f}  [{row["segmentation_provenance"]}]')
    print(f'identical segment lists: {a == b}')
    print(f'elapsed_s: {base.get("elapsed_s")} -> {cand.get("elapsed_s")}   timing_s: {cand.get("timing_s")}')
    ra, rb = base.get('raw_windows'), cand.get('raw_windows')
    if not ra or not rb:
        print('raw windows: not present in both files (run with --save-raw)')
        return 0
    ka = {(r['clip'], r['scope_id']): r for r in ra}
    kb = {(r['clip'], r['scope_id']): r for r in rb}
    common = sorted(set(ka) & set(kb))
    same_tokens = sum(ka[k]['input_tokens'] == kb[k]['input_tokens'] for k in common)
    same_output = sum(ka[k]['output'] == kb[k]['output'] for k in common)
    print(f'raw windows: baseline {len(ra)}, candidate {len(rb)}, in common {len(common)}')
    print(f'  same input token count: {same_tokens}/{len(common)}   identical output text: {same_output}/{len(common)}')
    shown = 0
    for k in common:
        if ka[k]['output'] != kb[k]['output'] and shown < args.show:
            shown += 1
            print(f'  window {k}: tokens {ka[k]["input_tokens"]} vs {kb[k]["input_tokens"]}\n    baseline : {ka[k]["output"][:300]}\n    candidate: {kb[k]["output"][:300]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
