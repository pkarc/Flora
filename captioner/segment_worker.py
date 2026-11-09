"""Long-running TempSeg-only worker: watches a folder for jobs and writes action segments (no captions).

Folder layout under --watch DIR (created on start):
  inbox/       drop job files here: NAME.json (write elsewhere, then rename into inbox/ so it appears atomically)
  processing/  the job being run (recovered into inbox/ if the worker died)
  out/         NAME.json results
  done/        finished job files        failed/   rejected job files + NAME.error.txt

Job file (relative paths are resolved against inbox/; absolute paths are fine):
  {"video": "main.mp4", "profile": "human"}
  {"video": "main.mp4", "left_wrist": "l.mp4", "right_wrist": "r.mp4", "profile": "yam", "yam32": false}
The job id is the file name without .json.

Backends: hf (the real model, loaded once) or fake (CPU, no weights; exercises everything except the model).
"""
from pathlib import Path
import argparse
import asyncio
import fcntl
import json
import math
import os
import re
import signal
import sys
import time
import traceback

from batch_utils.api import DataEnginePipeline
from batch_utils.bootstrap import production as a
from batch_utils.config import MODELS, PROFILE_POOLS
from batch_utils.media import frames_for
from batch_utils.models import runtime_event
from batch_utils.preprocessing import human_transport

JOB_ALLOWED = {'video', 'profile', 'left_wrist', 'right_wrist', 'yam32', 'instruction'}
JOB_ID = re.compile('[A-Za-z0-9][A-Za-z0-9_.-]{0,95}')
SCHEMA = 'flora_segments_v1'
TICK_S = 1 / 20


class HFBackend:
    """The real TempSeg model, loaded once. Same preprocessing and decoding as batch_utils.pipeline."""

    def load(self):
        from batch_utils.models import Session
        self.session = Session()
        self.model, self.proc = self.session.load('tempseg')

    def infer(self, window, frames, pool, roles, instruction):
        from batch_utils.models import generate
        from batch_utils.preprocessing import temporal_inputs
        inputs, _ = temporal_inputs(frames, window, pool, self.proc, roles, instruction)
        return generate(self.model, self.proc, inputs, 1024)


class FakeBackend:
    """Stands in for the model on CPU. Answers each window with the fragments of a known timeline of events.

    spec: {"events": [[start_s, end_s], ...], "garbage_windows": [window_index, ...]} (JSON file or None).
    Without a spec it uses a default timeline. The stitched result should reproduce the events.
    """

    def __init__(self, spec_path=None):
        spec = json.loads(Path(spec_path).read_text()) if spec_path else {}
        self.events = [tuple(e) for e in spec.get('events', [[3.0, 7.0], [14.0, 22.0], [40.0, 41.0]])]
        self.garbage = set(spec.get('garbage_windows', []))
        self.calls = 0

    def load(self):
        pass

    def infer(self, window, frames, pool, roles, instruction):
        self.calls += 1
        if window.window_index in self.garbage:
            return 'not json'
        origin_us = window.requested_view_start_us
        rows = []
        for start_s, end_s in self.events:
            s = round((start_s * 1e6 - origin_us) / 5e4)
            e = round((end_s * 1e6 - origin_us) / 5e4)
            lo, ro = s < window.valid_start_tick, e > window.valid_end_tick
            s, e = max(s, window.valid_start_tick), min(e, window.valid_end_tick)
            if s < e and s < window.core_end_tick and e > window.core_start_tick:
                rows.append((s, e, lo, ro))
        rows.sort()
        return json.dumps({'segments': [dict(s=s, e=e, lo=lo, ro=ro) for s, e, lo, ro in rows]}, separators=(',', ':'))


class Predictor:
    """Mirror of batch_utils.pipeline.Predictor with the model call behind a backend."""

    def __init__(self, backend, videos, source, pool, bound, instruction):
        self.backend, self.videos, self.source, self.pool, self.bound, self.instruction = backend, videos, source, pool, bound, instruction
        self.windows = 0

    def predict(self, request):
        from flora_runtime.video_segmentation.fine_subtask import FINE_RESPONSE_SCHEMA
        if request['source']['source_id'] != self.source.source_id or request['model_binding'] != self.bound:
            raise RuntimeError('Production request binding changed')
        predictions = []
        for payload in request['windows']:
            window = a.window_from_payload(payload, self.source)
            plan = a.temporal_frame_plan(window, list(self.videos['main'].pts_us))
            mosaic = 'Dex' if self.pool in ('dex', 'tianji') else 'YAM'
            images = frames_for(self.videos, set(plan['source_frame_indices']), mosaic=mosaic)
            if self.pool == 'human':
                images = human_transport(images)
            frames = [images[i] for i in plan['source_frame_indices']]
            text = self.backend.infer(window, frames, self.pool, list(self.videos), self.instruction)
            predictions.append(dict(scope_id=window.scope_id, prediction_json=text))
            self.windows += 1
        return dict(schema_version=FINE_RESPONSE_SCHEMA, source_id=self.source.source_id, model_io_contract=self.bound['model_io_contract'], model_binding=self.bound, predictions=predictions)


def segment_job(backend, job):
    videos, canonical = DataEnginePipeline._inputs(job['video'], job.get('profile', 'human'), job.get('left_wrist'), job.get('right_wrist'), job.get('yam32', False), job.get('instruction', ''))
    pool = 'yam32' if job.get('yam32') else PROFILE_POOLS[canonical]
    main = videos['main']
    source = a.AnnotationSource(source_id='upload_' + main.sha256[:24], location=str(main.path), duration_s=main.duration_s, fps=float(main.rate))
    bound = dict(candidate_id='flora_final', checkpoint_sha256=MODELS['tempseg']['sha256'], model_io_contract=a.fine_window_profile(a.PROFILES[pool]).model_io_contract)
    predictor = Predictor(backend, videos, source, pool, bound, job.get('instruction', ''))
    if len(videos) == 3:
        fine = a.FineSubtaskSegmenter(predictor=predictor, model_binding=bound, profile_id='robot_atomic_intent_v1', window_profile_id=a.PROFILES[pool], fail_on_unresolved=True, batch_concurrency=1, unresolved_stitch_tolerance_ticks=2, quality_policy=a.FineSegmentationQualityPolicy(drop_guard_only_segments=True))
        engine = a.HierarchicalSubtaskSegmenter(coarse_segmenter=a.NoCoarse(), fine_segmenter=fine, fine_prior_mode='none', fine_failure_policy='adaptive_fine_then_conservative_completion', fine_output_coverage_policy='allow_gaps', adaptive_fine_retry_max_windows=32)
    else:
        engine = a.production_segmenter(predictor, bound, pool)
    segments = asyncio.run(engine.segment_async(source))
    rows = [dict(start_s=s.span.start_s, end_s=s.span.end_s, segmentation_provenance=dict(s.metadata).get('segmentation_provenance', 'fine')) for s in segments]
    for r in rows:
        if not (math.isfinite(r['start_s']) and math.isfinite(r['end_s']) and 0 <= r['start_s'] < r['end_s'] <= main.duration_s + 0.001):
            raise RuntimeError('The pipeline returned an invalid segment interval.')
    return dict(schema_version=SCHEMA, duration_s=main.duration_s, profile=canonical, camera_roles=list(videos), segments=rows, windows=predictor.windows, model_revisions=dict(tempseg=MODELS['tempseg'], tempseg_base=MODELS['tempseg_base']), input_sha256={role: v.sha256 for role, v in videos.items()})


def read_job(path, base):
    row = json.loads(path.read_text())
    if not isinstance(row, dict) or set(row) - JOB_ALLOWED or 'video' not in row:
        raise ValueError('Invalid job file: allowed keys are ' + ', '.join(sorted(JOB_ALLOWED)) + ' and video is required.')
    for key in ('video', 'left_wrist', 'right_wrist'):
        if row.get(key) is not None:
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f'Invalid camera path for {key}.')
            value = Path(row[key]).expanduser()
            row[key] = value if value.is_absolute() else base / value
    return row


def write_atomic(path, payload):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


class Worker:

    def __init__(self, root, backend, settle_s=1.0):
        self.root, self.backend, self.settle_s = root, backend, settle_s
        self.dirs = {n: root / n for n in ('inbox', 'processing', 'out', 'done', 'failed')}
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        self.stop = False

    def recover(self):
        for p in sorted(self.dirs['processing'].glob('*.json')):
            os.replace(p, self.dirs['inbox'] / p.name)
            runtime_event('job_recovered', id=p.stem)

    def next_job(self):
        now = time.time()
        for p in sorted(self.dirs['inbox'].glob('*.json'), key=lambda p: (p.stat().st_mtime, p.name)):
            if now - p.stat().st_mtime >= self.settle_s:
                return p
        return None

    def run_one(self, inbox_path):
        name = inbox_path.stem
        claimed = self.dirs['processing'] / inbox_path.name
        os.replace(inbox_path, claimed)
        started = time.monotonic()
        out = self.dirs['out'] / (name + '.json')
        try:
            if not JOB_ID.fullmatch(name):
                raise ValueError('Job file name must match [A-Za-z0-9][A-Za-z0-9_.-]{0,95}.json')
            if out.exists():
                runtime_event('job_skipped_output_exists', id=name)
                os.replace(claimed, self.dirs['done'] / claimed.name)
                return True
            runtime_event('job_start', id=name)
            result = segment_job(self.backend, read_job(claimed, inbox_path.parent))
            result = dict(id=name, **result, elapsed_s=round(time.monotonic() - started, 2))
            write_atomic(out, result)
            os.replace(claimed, self.dirs['done'] / claimed.name)
            runtime_event('job_done', id=name, segments=len(result['segments']), windows=result['windows'], elapsed_s=result['elapsed_s'])
            return True
        except Exception as error:
            (self.dirs['failed'] / (name + '.error.txt')).write_text(f'{type(error).__name__}: {error}\n\n{traceback.format_exc()}')
            os.replace(claimed, self.dirs['failed'] / claimed.name)
            runtime_event('job_failed', id=name, error=f'{type(error).__name__}: {error}')
            return False

    def serve(self, once, poll_s):
        self.recover()
        runtime_event('worker_ready', watch=str(self.root), once=once)
        while not self.stop:
            job = self.next_job()
            if job is None:
                if once:
                    break
                time.sleep(poll_s)
                continue
            self.run_one(job)
        runtime_event('worker_stopped')


def main(argv=None):
    p = argparse.ArgumentParser(description='FLORA TempSeg-only worker: watches a folder and writes action segments without captions.')
    p.add_argument('--watch', type=Path, required=True, help='Work folder (inbox/, out/, ... are created inside).')
    p.add_argument('--backend', choices=['hf', 'fake'], default='hf')
    p.add_argument('--fake-spec', type=Path, help='fake backend: JSON {"events": [[s, e], ...], "garbage_windows": [i, ...]}.')
    p.add_argument('--once', action='store_true', help='Process what is in inbox/ and exit.')
    p.add_argument('--poll-interval', type=float, default=2.0)
    p.add_argument('--settle', type=float, default=1.0, help='Ignore job files modified less than this many seconds ago.')
    args = p.parse_args(argv)
    try:
        root = args.watch.expanduser().resolve()
        worker = Worker(root, FakeBackend(args.fake_spec) if args.backend == 'fake' else HFBackend(), args.settle)
        lock = open(root / '.worker.lock', 'w')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError(f'Another worker already uses {root}.')
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: setattr(worker, 'stop', True))
        worker.backend.load()
        worker.serve(args.once, args.poll_interval)
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f'FLORA worker failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
