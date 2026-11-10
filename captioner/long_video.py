"""Virtual clips for videos beyond the 120 s / 7200 frame limit, and seek-based frame decoding.

Nothing is re-encoded: a clip is a view onto frames [g0, g1) of the native file, with timestamps re-based so the
first frame is 0 (what the pipeline expects). Frames are decoded from the original file, so pixels are the native ones.
"""
from bisect import bisect_left
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import hashlib
import av
from batch_utils.config import MAX_DURATION_S, MAX_DIMENSION, MAX_FRAME_COUNT
from batch_utils.media import InputError


@dataclass(frozen=True)
class Index:
    path: Path
    sha256: str
    rate: Fraction
    time_base: Fraction
    raw_pts: tuple
    pts_us: tuple
    last_length_us: object
    size: tuple

    @property
    def duration_us(self):
        return round(self.pts_us[-1] + (self.last_length_us or Fraction(1000000) / self.rate))


@dataclass(frozen=True)
class Clip:
    """Duck-types batch_utils.media.Video, plus what the seek decoder needs."""
    path: Path
    sha256: str
    rate: Fraction
    pts_us: tuple
    duration_us: int
    size: tuple
    cfr: bool
    raw_pts: tuple
    time_base: Fraction
    start_s: float
    index: int

    @property
    def duration_s(self):
        return self.duration_us / 1000000


def index_video(path):
    """Whole-file timestamp index from packets (no pixel decoding); same validation as media.probe."""
    with av.open(str(path), options={'err_detect': 'explode'}) as container:
        if len(container.streams.video) != 1:
            raise InputError('Use a file containing one video stream.')
        stream = container.streams.video[0]
        rate = stream.average_rate
        if not rate or rate < 4:
            raise InputError('The input needs at least 4 source frames per second.')
        size = (stream.codec_context.width, stream.codec_context.height)
        if min(size) <= 0 or max(size) > MAX_DIMENSION:
            raise InputError('Video dimensions must be at most 4096 pixels per side.')
        packets = []
        for packet in container.demux(stream):
            if packet.size == 0:
                continue
            if packet.pts is None:
                raise InputError('Every source frame needs a presentation timestamp.')
            packets.append((packet.pts, packet.duration))
        time_base = stream.time_base
    with av.open(str(path), options={'err_detect': 'explode'}) as container:
        container.streams.video[0].codec_context.thread_count = 1
        # frames hidden by an edit list are in the packets but never decoded: zero is the first decoded frame
        first = next(iter(container.decode(container.streams.video[0])), None)
    if first is None or first.pts is None:
        raise InputError('Every source frame needs a presentation timestamp.')
    packets = sorted((p for p in packets if p[0] >= first.pts), key=lambda p: p[0])
    if len(packets) < 2:
        raise InputError('The video contains too few source frames.')
    zero = first.pts
    raw = tuple(p[0] for p in packets)
    pts_us = tuple(Fraction(r - zero) * time_base * 1000000 for r in raw)
    if any(b <= a for a, b in zip(pts_us, pts_us[1:])):
        raise InputError('Source timestamps must be strictly increasing.')
    last = packets[-1][1]
    last_length = Fraction(last) * time_base * 1000000 if last else None
    with Path(path).open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return Index(Path(path), digest, rate, time_base, raw, pts_us, last_length, size)


def make_clip(ix, g0, g1, number):
    pts = ix.pts_us[g0:g1]
    rel = tuple(t - pts[0] for t in pts)
    if len(rel) < 2:
        raise InputError('The video contains too few source frames.')
    tail = ix.last_length_us if g1 >= len(ix.pts_us) else None
    duration_us = round(rel[-1] + (tail or Fraction(1000000) / ix.rate))
    if duration_us > MAX_DURATION_S * 1000000 or len(rel) > MAX_FRAME_COUNT:
        raise InputError(f'Clip exceeds {MAX_DURATION_S:g} seconds or {MAX_FRAME_COUNT} frames.')
    cfr = all(abs(t - Fraction(i * 1000000) / ix.rate) <= 1 for i, t in enumerate(rel))
    return Clip(ix.path, ix.sha256, ix.rate, rel, duration_us, ix.size, cfr, ix.raw_pts[g0:g1], ix.time_base, float(pts[0]) / 1000000, number)


def fits_directly(ix, direct_max_s):
    limit = min(direct_max_s, MAX_DURATION_S)
    return ix.duration_us <= limit * 1000000 and len(ix.pts_us) <= MAX_FRAME_COUNT and ix.pts_us[-1] < limit * 1000000


def plan_clips(indexes, clip_s, overlap_s, direct_max_s):
    """Per clip: {role: Clip}. One clip when the video fits the limits; otherwise overlapping clips."""
    main = indexes['main']
    for role, ix in indexes.items():
        if ix.rate != main.rate or ix.pts_us != main.pts_us:
            raise InputError('Camera files must have identical relative frame timestamps, frame counts and duration.')
    n = len(main.pts_us)
    if fits_directly(main, direct_max_s):
        return [{role: make_clip(ix, 0, n, 0) for role, ix in indexes.items()}]
    clip_s = min(clip_s, 0.97 * MAX_FRAME_COUNT / float(main.rate), 0.97 * MAX_DURATION_S)
    if overlap_s <= 0 or overlap_s >= clip_s / 2:
        raise ValueError('overlap must be positive and less than half the clip length.')
    times = [float(t) for t in main.pts_us]
    step_us, clip_us = (clip_s - overlap_s) * 1000000, clip_s * 1000000
    clips, k = [], 0
    while True:
        a = k * step_us
        g0 = bisect_left(times, a)
        g1 = bisect_left(times, a + clip_us)
        clips.append({role: make_clip(ix, g0, g1, k) for role, ix in indexes.items()})
        if g1 >= n:
            break
        k += 1
    return clips


def decode_frames(clip, wanted, canonical=False, seek=True):
    """Same pixels as media.decode_selected, but seeks to a keyframe before the first wanted frame."""
    wanted = set(wanted)
    if not wanted:
        return {}
    if min(wanted) < 0 or max(wanted) >= len(clip.raw_pts):
        raise InputError('A requested source frame is outside the video.')
    targets = {clip.raw_pts[i]: i for i in wanted}
    result, last = {}, max(targets)
    with av.open(str(clip.path), options={'err_detect': 'explode'}) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        if seek:
            margin = int(2 / clip.time_base)
            container.seek(max(0, min(targets) - margin), stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts in targets:
                use_canonical = canonical and frame.format.name == 'yuv420p' and (frame.width % 16 == 0) and (frame.height % 2 == 0) and (int(frame.colorspace) in (2, 5, 6)) and (int(frame.color_range) in (0, 1))
                if use_canonical:
                    from canonical_rgb import canonical_rgb
                    result[targets[frame.pts]] = canonical_rgb(frame)
                else:
                    result[targets[frame.pts]] = frame.to_image().convert('RGB')
            if len(result) == len(wanted) or (frame.pts is not None and frame.pts > last):
                break
    if set(result) != wanted:
        raise InputError('A requested source frame could not be decoded.')
    return result


def frames_for(videos, wanted, mosaic='YAM', seek=True):
    frames = {role: decode_frames(video, wanted, canonical=len(videos) == 3, seek=seek) for role, video in videos.items()}
    if len(videos) == 1:
        return frames['main']
    from composers import compose_top_dominant_mosaic, compose_frame
    if mosaic == 'YAM':
        return {i: compose_top_dominant_mosaic(frames['main'][i], frames['left_wrist'][i], frames['right_wrist'][i]) for i in wanted}
    if mosaic == 'Dex':
        return {i: compose_frame(dict(ego=frames['main'][i], left_wrist=frames['left_wrist'][i], right_wrist=frames['right_wrist'][i]), ('ego', 'left_wrist', 'right_wrist')) for i in wanted}
    raise InputError('Choose a supported camera layout.')


def _cut_point(left, right, a, b, tol, margin):
    """Cut time inside the overlap [a, b]: where both clips agree on a boundary, else the center."""
    lo, hi = a + margin, b - margin
    center = (a + b) / 2
    lb = [t for s in left for t in (s['start_s'], s['end_s']) if lo <= t <= hi]
    rb = [t for s in right for t in (s['start_s'], s['end_s']) if lo <= t <= hi]
    pairs = [((x + y) / 2, abs(x - y)) for x in lb for y in rb if abs(x - y) <= tol]
    if pairs:
        return min(pairs, key=lambda p: (abs(p[0] - center), p[1]))[0], 'matched_boundary'
    return center, 'center'


def _tag(segment, suffix):
    base = segment['segmentation_provenance']
    return dict(segment, segmentation_provenance=base if suffix in base else base + '+' + suffix)


def merge_clips(clip_segments, spans, tol=0.5, margin=4.0):
    """Join per-clip segments (global seconds) into one list. spans: [(start_s, end_s)] per clip, overlapping.

    At each overlap, cut where both clips agree on a boundary (else the middle). Segments ending before the cut come
    from the earlier clip, those starting after it from the later one, and segments crossing it in both clips are
    united (earlier start, later end). Returns (segments, info per boundary).
    """
    final, info = [], []
    current = sorted(clip_segments[0], key=lambda s: (s['start_s'], s['end_s']))
    for k in range(1, len(clip_segments)):
        a, b = spans[k][0], spans[k - 1][1]
        right = sorted(clip_segments[k], key=lambda s: (s['start_s'], s['end_s']))
        cut, method = _cut_point(current, right, a, b, tol, margin)
        keep_left = [s for s in current if s['end_s'] <= cut + tol]
        cross_left = [s for s in current if s['start_s'] < cut - tol and s['end_s'] > cut + tol]
        keep_right = [s for s in right if s['start_s'] >= cut - tol and s['end_s'] > cut + tol]
        cross_right = [s for s in right if s['start_s'] < cut - tol and s['end_s'] > cut + tol]
        dropped = len(current) - len(keep_left) - len(cross_left)
        merged, unmatched, used = [], 0, set()
        for s in cross_left:
            best, best_overlap = None, 1.0
            for j, r in enumerate(cross_right):
                overlap = min(s['end_s'], r['end_s']) - max(s['start_s'], r['start_s'])
                if j not in used and overlap > best_overlap:
                    best, best_overlap = j, overlap
            if best is None:
                keep_left.append(_tag(s, 'clip_unmatched'))
                unmatched += 1
            else:
                used.add(best)
                merged.append(_tag(dict(s, end_s=max(s['end_s'], cross_right[best]['end_s'])), 'clip_merge'))
        for j, r in enumerate(cross_right):
            if j not in used:
                merged.append(_tag(r, 'clip_unmatched'))
                unmatched += 1
        final.extend(keep_left)
        current = sorted(merged + keep_right, key=lambda s: (s['start_s'], s['end_s']))
        info.append(dict(boundary=k, overlap_s=[round(a, 3), round(b, 3)], cut_s=round(cut, 3), method=method, merged=len(used), unmatched=unmatched, dropped_edge_segments=dropped))
    final.extend(current)
    return sorted(final, key=lambda s: (s['start_s'], s['end_s'])), info
