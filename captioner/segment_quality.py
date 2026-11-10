"""Quality summary for a segment-worker result: provenance counts and the segments worth a manual look (stdlib only).

  python captioner/segment_quality.py out/t01.json

These are heuristics on how each segment was produced, not calibrated confidences:
  fallback_provenance  the pipeline could not resolve the segment normally (fine_conservative_completion) or two
                       clips disagreed at a cut (clip_unmatched)
  similar_to_N         overlaps segment N with IoU >= 0.5: often a duplicate left by the retry passes, though
                       genuinely concurrent actions look the same
Also counted: segments repaired by a retry or joined across clips, and small uncovered gaps between segments.
The worker writes the same summary under "quality" in every result.
"""
from collections import Counter
import json
import sys

FALLBACK = ('conservative_completion', 'clip_unmatched')
REPAIRED = ('repaired', 'clip_merge')


def _iou(x, y):
    inter = max(0.0, min(x['end_s'], y['end_s']) - max(x['start_s'], y['start_s']))
    union = (x['end_s'] - x['start_s']) + (y['end_s'] - y['start_s']) - inter
    return inter / union if union > 0 else 0.0


def summarize(rows, similar_iou=0.5, max_gap_s=1.0):
    order = sorted(range(len(rows)), key=lambda i: (rows[i]['start_s'], rows[i]['end_s']))
    reasons = {}
    for i, row in enumerate(rows):
        if any(k in row['segmentation_provenance'] for k in FALLBACK):
            reasons.setdefault(i, []).append('fallback_provenance')
    similar = 0
    for a, i in enumerate(order):
        for j in order[a + 1:]:
            if rows[j]['start_s'] >= rows[i]['end_s']:
                break
            if _iou(rows[i], rows[j]) >= similar_iou:
                similar += 1
                reasons.setdefault(i, []).append(f'similar_to_{j}')
                reasons.setdefault(j, []).append(f'similar_to_{i}')
    gaps, covered = [], None
    for i in order:
        if covered is not None and 1e-6 < rows[i]['start_s'] - covered <= max_gap_s:
            gaps.append([round(covered, 3), round(rows[i]['start_s'], 3)])
        covered = rows[i]['end_s'] if covered is None else max(covered, rows[i]['end_s'])
    provenance = Counter(r['segmentation_provenance'] for r in rows)
    return dict(
        total=len(rows),
        provenance=dict(provenance),
        repaired=sum(n for k, n in provenance.items() if any(m in k for m in REPAIRED)),
        fallback=sum(n for k, n in provenance.items() if any(m in k for m in FALLBACK)),
        similar_pairs=similar,
        small_gaps=gaps,
        flagged=[dict(index=i, start_s=rows[i]['start_s'], end_s=rows[i]['end_s'], segmentation_provenance=rows[i]['segmentation_provenance'], reasons=reasons[i]) for i in sorted(reasons)],
    )


def main(argv=None):
    if len(argv if argv is not None else sys.argv[1:]) != 1:
        print(__doc__)
        return 2
    path = (argv if argv is not None else sys.argv[1:])[0]
    result = json.load(open(path))
    q = summarize(result['segments'])
    print(f'{q["total"]} segments | repaired/merged {q["repaired"]} | fallback {q["fallback"]} | similar pairs {q["similar_pairs"]} | small gaps {len(q["small_gaps"])}')
    print('provenance:', q['provenance'])
    if q['small_gaps']:
        print('small gaps (s):', ', '.join(f'{a}-{b}' for a, b in q['small_gaps']))
    for f in q['flagged']:
        print(f'  #{f["index"]:>3}  {f["start_s"]:8.2f} -> {f["end_s"]:8.2f}  [{f["segmentation_provenance"]}]  {", ".join(f["reasons"])}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
