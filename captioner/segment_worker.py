"""Long-running TempSeg-only worker: watches a folder for jobs and writes action segments (no captions).

Folder layout under --watch DIR (created on start):
  inbox/       drop job files here: NAME.json (write elsewhere, then rename into inbox/ so it appears atomically)
  processing/  the job being run (recovered into inbox/ if the worker died)
  out/         NAME.json results
  done/        finished job files        failed/   rejected job files + NAME.error.txt

Videos longer than 120 s are cut into overlapping virtual clips (no re-encoding) and the segments merged.

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
import threading
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from batch_utils.api import _PROFILES
from batch_utils.bootstrap import production as a
from batch_utils.config import MODELS, PROFILE_POOLS
from batch_utils.media import InputError, check_synchronized, frames_for as reference_frames_for, probe
from batch_utils.models import runtime_event
from batch_utils.preprocessing import human_transport
import long_video as lv
from segment_quality import summarize as summarize_quality

JOB_ALLOWED = {'video', 'profile', 'left_wrist', 'right_wrist', 'yam32', 'instruction'}
JOB_ID = re.compile('[A-Za-z0-9][A-Za-z0-9_.-]{0,95}')
SCHEMA = 'flora_segments_v1'


class HFBackend:
    """The real TempSeg model, loaded once. Same preprocessing and decoding as batch_utils.pipeline."""

    def load(self):
        from batch_utils.models import Session
        self.session = Session()
        self.model, self.proc = self.session.load('tempseg')

    def prepare(self, window, frames, pool, roles, instruction):
        """CPU only (processor); safe to run while the GPU generates another window."""
        from batch_utils.preprocessing import temporal_inputs
        inputs, _ = temporal_inputs(frames, window, pool, self.proc, roles, instruction)
        return inputs

    def infer(self, window, prepared):
        from batch_utils.models import generate
        self.last_input_tokens = [int(prepared['input_ids'].shape[1])]
        return generate(self.model, self.proc, prepared, 1024)


class FakeBackend:
    """Stands in for the model on CPU. Answers each window with the fragments of a known timeline of events.

    spec: {"events": [[start_s, end_s], ...], "garbage_windows": [window_index, ...]} (JSON file or None).
    Without a spec it uses a default timeline. The stitched result should reproduce the events.
    """

    def __init__(self, spec_path=None):
        spec = json.loads(Path(spec_path).read_text()) if spec_path else {}
        self.events = [tuple(e) for e in spec.get('events', [[3.0, 7.0], [14.0, 22.0], [40.0, 41.0]])]
        self.garbage = set(spec.get('garbage_windows', []))
        self.delay_s = float(spec.get('infer_delay_s', 0))
        if not spec.get('batch'):
            self.infer_batch = None
        self.calls = 0
        self.offset_s = 0.0

    def load(self):
        pass

    def begin_clip(self, offset_s):
        self.offset_s = offset_s

    def prepare(self, window, frames, pool, roles, instruction):
        return None

    def infer_batch(self, windows, prepared):
        """Only used when the spec sets "batch": true, to exercise the batched path of the Predictor."""
        texts = [self.infer(w, p) for w, p in zip(windows, prepared)]
        self.last_input_tokens = [1000 + w.window_index for w in windows]
        return texts

    def infer(self, window, prepared):
        self.calls += 1
        time.sleep(self.delay_s)
        if window.window_index in self.garbage:
            return 'not json'
        origin_us = window.requested_view_start_us
        rows = []
        for start_s, end_s in self.events:
            s = round(((start_s - self.offset_s) * 1e6 - origin_us) / 5e4)
            e = round(((end_s - self.offset_s) * 1e6 - origin_us) / 5e4)
            # the model cannot see past the video's own ends: nothing continues beyond them
            lo = s < window.valid_start_tick and window.valid_start_us > 0
            ro = e > window.valid_end_tick and window.valid_end_us < window.video_end_us
            s, e = max(s, window.valid_start_tick), min(e, window.valid_end_tick)
            if s < e and s < window.core_end_tick and e > window.core_start_tick:
                rows.append((s, e, lo, ro))
        rows.sort()
        return json.dumps({'segments': [dict(s=s, e=e, lo=lo, ro=ro) for s, e, lo, ro in rows]}, separators=(',', ':'))


class Predictor:
    """Mirror of batch_utils.pipeline.Predictor with the model call behind a backend and seek-based decoding.

    With prefetch=1 the next window is decoded and preprocessed in a thread while the current one is generated.
    Each window is prepared and generated exactly as before; only the order of the work changes.
    """

    def __init__(self, backend, videos, source, pool, bound, instruction, fast=True, verify=0, prefetch=1, batch_size=1, threads=1):
        self.backend, self.videos, self.source, self.pool, self.bound, self.instruction = backend, videos, source, pool, bound, instruction
        self.fast, self.verify, self.prefetch, self.batch_size, self.threads = fast, verify, prefetch, batch_size, threads
        self.windows = 0
        self.raw = []
        self.timing = dict(decode=0.0, transport=0.0, processor=0.0, generate=0.0, wait=0.0)
        self._lock = threading.Lock()

    def _add(self, key, started):
        with self._lock:
            self.timing[key] += time.monotonic() - started

    def _frames(self, wanted, mosaic):
        started = time.monotonic()
        images = lv.frames_for(self.videos, wanted, mosaic=mosaic) if self.fast else reference_frames_for(self.videos, wanted, mosaic=mosaic)
        self._add('decode', started)
        with self._lock:
            check = self.fast and self.verify > 0
            self.verify -= check
        if check:
            if all(getattr(v, 'start_s', 0) == 0 for v in self.videos.values()):
                slow = reference_frames_for(self.videos, wanted, mosaic=mosaic)
            else:
                slow = lv.frames_for(self.videos, wanted, mosaic=mosaic, seek=False)
            if set(slow) != set(images) or any(slow[i].size != images[i].size or slow[i].tobytes() != images[i].tobytes() for i in images):
                raise RuntimeError('Seek-based decoding differs from sequential decoding; rerun with --no-fast-decode.')
            runtime_event('decode_verified', frames=len(images))
        return images

    def _prepare(self, payload):
        window = a.window_from_payload(payload, self.source)
        plan = a.temporal_frame_plan(window, list(self.videos['main'].pts_us))
        mosaic = 'Dex' if self.pool in ('dex', 'tianji') else 'YAM'
        images = self._frames(set(plan['source_frame_indices']), mosaic)
        if self.pool == 'human':
            started = time.monotonic()
            images = human_transport(images)
            self._add('transport', started)
        frames = [images[i] for i in plan['source_frame_indices']]
        started = time.monotonic()
        prepared = self.backend.prepare(window, frames, self.pool, list(self.videos), self.instruction)
        self._add('processor', started)
        return window, prepared

    def _prepared(self, payloads):
        if self.prefetch <= 0:
            for payload in payloads:
                started = time.monotonic()
                result = self._prepare(payload)
                self._add('wait', started)
                yield result
            return
        pending, iterator = deque(), iter(payloads)
        with ThreadPoolExecutor(max_workers=max(1, self.threads)) as executor:
            try:
                while True:
                    while len(pending) < self.prefetch:
                        payload = next(iterator, None)
                        if payload is None:
                            break
                        pending.append(executor.submit(self._prepare, payload))
                    if not pending:
                        return
                    started = time.monotonic()
                    result = pending.popleft().result()
                    self._add('wait', started)
                    # queue the next window before the caller starts generating this one
                    payload = next(iterator, None)
                    if payload is not None:
                        pending.append(executor.submit(self._prepare, payload))
                    yield result
            finally:
                for future in pending:
                    future.cancel()

    def predict(self, request):
        from flora_runtime.video_segmentation.fine_subtask import FINE_RESPONSE_SCHEMA
        if request['source']['source_id'] != self.source.source_id or request['model_binding'] != self.bound:
            raise RuntimeError('Production request binding changed')
        predictions, chunk = [], []
        batched = callable(getattr(self.backend, 'infer_batch', None))
        size = self.batch_size if batched else 1

        def flush():
            if not chunk:
                return
            started = time.monotonic()
            if batched:
                texts = self.backend.infer_batch([w for w, _ in chunk], [p for _, p in chunk])
            else:
                texts = [self.backend.infer(*chunk[0])]
            self._add('generate', started)
            tokens = getattr(self.backend, 'last_input_tokens', None) or [None] * len(chunk)
            for (window, _), text, n in zip(chunk, texts, tokens):
                predictions.append(dict(scope_id=window.scope_id, prediction_json=text))
                self.raw.append(dict(scope_id=window.scope_id, window_index=window.window_index, view_start_s=window.requested_view_start_us / 1000000, input_tokens=n, output=text))
            self.windows += len(chunk)
            chunk.clear()

        for item in self._prepared(request['windows']):
            chunk.append(item)
            if len(chunk) >= size:
                flush()
        flush()
        return dict(schema_version=FINE_RESPONSE_SCHEMA, source_id=self.source.source_id, model_io_contract=self.bound['model_io_contract'], model_binding=self.bound, predictions=predictions)


def job_arguments(job):
    """Same argument checks as DataEnginePipeline._inputs, without probing the files."""
    profile, left, right = job.get('profile', 'human'), job.get('left_wrist'), job.get('right_wrist')
    yam32, instruction = job.get('yam32', False), job.get('instruction', '')
    if not isinstance(profile, str) or profile.lower() not in _PROFILES:
        raise InputError('profile must be human, yam, dex, or tianji')
    canonical = _PROFILES[profile.lower()]
    if bool(left) != bool(right):
        raise InputError('Provide both left and right wrist videos together.')
    if canonical == 'Human' and left:
        raise InputError('The human profile accepts one video.')
    if not isinstance(yam32, bool):
        raise InputError('yam32 must be true or false.')
    if yam32 and (canonical != 'YAM' or not left):
        raise InputError('yam32 requires YAM and all three camera views.')
    if not isinstance(instruction, str) or len(instruction.strip()) > 2000:
        raise InputError('instruction must be a string of at most 2000 characters.')
    paths = {'main': job['video']}
    if left:
        paths.update(left_wrist=left, right_wrist=right)
    resolved = {}
    for role, value in paths.items():
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise InputError('Each camera path must be a nonempty local path.')
        path = Path(value).expanduser().resolve(strict=True)
        if not path.is_file() or path.stat().st_size <= 0:
            raise InputError('Each input must be a nonempty local video file.')
        resolved[role] = path
    return canonical, resolved, yam32, instruction


def segment_clip(backend, videos, pool, instruction, fast, verify, prefetch=1, batch_size=1, threads=1):
    main = videos['main']
    source = a.AnnotationSource(source_id='upload_' + main.sha256[:24], location=str(main.path), duration_s=main.duration_s, fps=float(main.rate))
    bound = dict(candidate_id='flora_final', checkpoint_sha256=MODELS['tempseg']['sha256'], model_io_contract=a.fine_window_profile(a.PROFILES[pool]).model_io_contract)
    predictor = Predictor(backend, videos, source, pool, bound, instruction, fast, verify, prefetch, batch_size, threads)
    if len(videos) == 3:
        fine = a.FineSubtaskSegmenter(predictor=predictor, model_binding=bound, profile_id='robot_atomic_intent_v1', window_profile_id=a.PROFILES[pool], fail_on_unresolved=True, batch_concurrency=1, unresolved_stitch_tolerance_ticks=2, quality_policy=a.FineSegmentationQualityPolicy(drop_guard_only_segments=True))
        engine = a.HierarchicalSubtaskSegmenter(coarse_segmenter=a.NoCoarse(), fine_segmenter=fine, fine_prior_mode='none', fine_failure_policy='adaptive_fine_then_conservative_completion', fine_output_coverage_policy='allow_gaps', adaptive_fine_retry_max_windows=32)
    else:
        engine = a.production_segmenter(predictor, bound, pool)
    segments = asyncio.run(engine.segment_async(source))
    rows = [dict(start_s=s.span.start_s, end_s=s.span.end_s, segmentation_provenance=dict(s.metadata).get('segmentation_provenance', 'fine')) for s in segments]
    return rows, predictor


def segment_job(backend, job, opts):
    canonical, paths, yam32, instruction = job_arguments(job)
    pool = 'yam32' if yam32 else PROFILE_POOLS[canonical]
    indexes = {role: lv.index_video(path) for role, path in paths.items()}
    plans = lv.plan_clips(indexes, opts.clip_s, opts.overlap_s, opts.direct_max_s)
    fast = not opts.no_fast_decode
    if len(plans) == 1:
        # one clip = the whole video: confirm our packet index equals the reference probe, else use the reference path
        reference = {role: probe(path) for role, path in paths.items()}
        if not all(reference[r].pts_us == plans[0][r].pts_us and reference[r].duration_us == plans[0][r].duration_us for r in reference):
            runtime_event('index_differs_from_probe', note='using the reference decoder for this job')
            plans, fast = [reference], False
    for videos in plans:
        check_synchronized(videos)
    main_index = indexes['main']
    duration_s = main_index.duration_us / 1000000
    per_clip, spans, windows, raw = [], [], 0, []
    timing = dict(decode=0.0, transport=0.0, processor=0.0, generate=0.0, wait=0.0)
    for number, videos in enumerate(plans):
        offset = getattr(videos['main'], 'start_s', 0.0)
        if hasattr(backend, 'begin_clip'):
            backend.begin_clip(offset)
        try:
            rows, predictor = segment_clip(backend, videos, pool, instruction, fast, opts.verify_decode if number == 0 else 0, opts.prefetch, opts.vllm_batch, opts.prep_threads)
        except Exception as error:
            if len(plans) == 1:
                raise
            raise RuntimeError(f'clip {number + 1}/{len(plans)} ({offset:.1f}-{offset + videos["main"].duration_s:.1f} s) failed: {type(error).__name__}: {error}') from error
        windows += predictor.windows
        raw.extend(dict(r, clip=number, view_start_s=round(r['view_start_s'] + offset, 6)) for r in predictor.raw)
        for key, value in predictor.timing.items():
            timing[key] += value
        per_clip.append([dict(r, start_s=round(r['start_s'] + offset, 6), end_s=round(r['end_s'] + offset, 6)) for r in rows])
        spans.append((offset, offset + videos['main'].duration_s))
        if len(plans) > 1:
            runtime_event('clip_done', clip=number + 1, of=len(plans), segments=len(rows), windows=predictor.windows)
    if len(plans) == 1:
        rows, merge = per_clip[0], []
    else:
        rows, merge = lv.merge_clips(per_clip, spans)
    for r in rows:
        if not (math.isfinite(r['start_s']) and math.isfinite(r['end_s']) and 0 <= r['start_s'] < r['end_s'] <= duration_s + 0.001):
            raise RuntimeError('The pipeline returned an invalid segment interval.')
    result = dict(schema_version=SCHEMA, duration_s=duration_s, profile=canonical, camera_roles=list(paths), segments=rows, quality=summarize_quality(rows), windows=windows, decode_s=round(timing['decode'], 2), timing_s={k: round(v, 2) for k, v in timing.items()}, prefetch=opts.prefetch, decoder='seek' if fast else 'reference')
    if opts.save_raw:
        result['raw_windows'] = raw
    if len(plans) > 1:
        result.update(clips=[dict(index=i, start_s=round(a_, 3), end_s=round(b_, 3)) for i, (a_, b_) in enumerate(spans)], merge=merge)
    return dict(result, model_revisions=dict(tempseg=MODELS['tempseg'], tempseg_base=MODELS['tempseg_base']), input_sha256={role: ix.sha256 for role, ix in indexes.items()})


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

    def __init__(self, root, backend, opts):
        self.root, self.backend, self.opts, self.settle_s = root, backend, opts, opts.settle
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
            result = segment_job(self.backend, read_job(claimed, inbox_path.parent), self.opts)
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


def make_backend(args):
    if args.backend == 'fake':
        return FakeBackend(args.fake_spec)
    if args.backend == 'vllm':
        from vllm_backend import VLLMBackend
        return VLLMBackend(args.vllm_gpu_util, args.vllm_max_len, args.vllm_lora_dtype, args.vllm_eager)
    return HFBackend()


def main(argv=None):
    p = argparse.ArgumentParser(description='FLORA TempSeg-only worker: watches a folder and writes action segments without captions.')
    p.add_argument('--watch', type=Path, required=True, help='Work folder (inbox/, out/, ... are created inside).')
    p.add_argument('--backend', choices=['hf', 'vllm', 'fake'], default='hf')
    p.add_argument('--fake-spec', type=Path, help='fake backend: JSON {"events": [[s, e], ...], "garbage_windows": [i, ...]}.')
    p.add_argument('--once', action='store_true', help='Process what is in inbox/ and exit.')
    p.add_argument('--poll-interval', type=float, default=2.0)
    p.add_argument('--settle', type=float, default=1.0, help='Ignore job files modified less than this many seconds ago.')
    p.add_argument('--clip-s', type=float, default=100.0, help='Videos beyond 120 s / 7200 frames are cut into virtual clips of this length (no re-encoding).')
    p.add_argument('--overlap-s', type=float, default=20.0, help='Overlap between consecutive clips; segments are merged inside it.')
    p.add_argument('--direct-max-s', type=float, default=120.0, help='Videos up to this length run as one clip (cannot exceed 120). Lower it to test clipping.')
    p.add_argument('--verify-decode', type=int, default=0, metavar='N', help='Compare seek-based decoding with sequential decoding on the first N windows of each job.')
    p.add_argument('--prefetch', type=int, default=1, help='Windows prepared ahead on the CPU while the GPU generates (0 = sequential, as the reference pipeline).')
    p.add_argument('--prep-threads', type=int, default=1, help='Threads preparing windows (decode, JPEG, processor). Use with --prefetch >= the batch size.')
    p.add_argument('--save-raw', action='store_true', help='Store each window\'s raw model output and input token count in the result (for comparing backends).')
    p.add_argument('--vllm-batch', type=int, default=8, help='vllm backend: windows generated together.')
    p.add_argument('--vllm-gpu-util', type=float, default=0.85)
    p.add_argument('--vllm-max-len', type=int, default=16384)
    p.add_argument('--vllm-lora-dtype', default='auto', choices=['auto', 'float16', 'bfloat16'], help='vLLM cannot hold the adapter in float32 (the reference does), so this path always differs numerically.')
    p.add_argument('--vllm-eager', action='store_true', help='Disable CUDA graphs.')
    p.add_argument('--no-fast-decode', action='store_true', help='Use the reference sequential decoder (slow).')
    args = p.parse_args(argv)
    try:
        root = args.watch.expanduser().resolve()
        worker = Worker(root, make_backend(args), args)
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
