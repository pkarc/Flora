"""Preview action segments on a video using only ffmpeg/ffprobe (no GPU, no model, no Flora environment).

  python captioner/segments_preview.py --video v.mp4 --segments out/t01.json --out preview/

Writes into --out:
  marked.mp4            the video with a label per active segment, a timeline bar with a moving cursor,
                        and a red border flash at the start of every segment (re-encoded; for review only)
  frames/segNN_a_start.png   first frame of the segment (t = start_s)
  frames/segNN_b_end.png     last frame of the segment (end_s is exclusive, so one frame before it)
  frames/segNN_c_after.png   first frame after the segment, to see the change
  contact_sheet.png     one row per segment: start frame on the left, end frame on the right
and prints a table of the segments. Accepts the worker's output and infer.py's output (any JSON with
"segments": [{"start_s", "end_s"}, ...]).
"""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys

FONTS = ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf',
         '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf', '/Library/Fonts/Arial Bold.ttf',
         '/System/Library/Fonts/Supplemental/Arial Bold.ttf', 'C:/Windows/Fonts/arialbd.ttf']
COLORS = ['0x3b82f6', '0x10b981', '0xf59e0b', '0xec4899', '0x8b5cf6', '0x06b6d4']


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f'{cmd[0]} failed: {r.stderr.strip()[-600:]}')
    return r.stdout


def probe(video):
    out = run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,avg_frame_rate:format=duration', '-of', 'json', str(video)])
    info = json.loads(out)
    stream = info['streams'][0]
    num, den = stream['avg_frame_rate'].split('/')
    return int(stream['width']), int(stream['height']), int(num) / int(den), float(info['format']['duration'])


def load_segments(path, duration):
    rows = json.loads(Path(path).read_text())['segments']
    segments = []
    for i, r in enumerate(rows, 1):
        s, e = float(r['start_s']), float(r['end_s'])
        if not 0 <= s < e <= duration + 0.05:
            raise ValueError(f'segment {i} [{s}, {e}] lies outside the video (duration {duration:.3f} s)')
        segments.append((s, min(e, duration)))
    if not segments:
        raise ValueError('The file contains no segments.')
    return segments


def lanes(segments):
    """Greedy lane per segment so overlapping segments get separate label rows."""
    ends, result = [], []
    for s, e in segments:
        for lane, end in enumerate(ends):
            if s >= end:
                ends[lane] = e
                result.append(lane)
                break
        else:
            ends.append(e)
            result.append(len(ends) - 1)
    return result


def extract_frames(video, segments, fps, duration, out):
    frames = out / 'frames'
    frames.mkdir(parents=True, exist_ok=True)
    last = max(0.0, duration - 1 / fps)
    paths = []
    for i, (s, e) in enumerate(segments, 1):
        for tag, t in (('a_start', s), ('b_end', max(s, e - 1 / fps)), ('c_after', min(e, last))):
            path = frames / f'seg{i:02d}_{tag}.png'
            run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{min(t, last):.4f}', '-i', str(video), '-frames:v', '1', str(path)])
            if tag != 'c_after':
                paths.append(path)
    return paths


def contact_sheet(paths, out):
    listing = out / 'contact_sheet.txt'
    listing.write_text(''.join(f"file '{p.resolve()}'\nduration 1\n" for p in paths) + f"file '{paths[-1].resolve()}'\n")
    rows = (len(paths) + 1) // 2
    run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', str(listing), '-vf', f'scale=640:-2,tile=2x{rows}:padding=6:color=black', '-frames:v', '1', str(out / 'contact_sheet.png')])
    listing.unlink()


def marked_video(video, segments, size, duration, font, out):
    w, h = size
    n = len(segments)
    fs = max(16, h // 28)
    bar_h = max(10, h // 45)
    bar_y = h - bar_h - 6
    parts = []
    for i, ((s, e), lane) in enumerate(zip(segments, lanes(segments)), 1):
        color = COLORS[(i - 1) % len(COLORS)]
        x0, x1 = round(s / duration * w), round(e / duration * w)
        parts.append(f'drawbox=x={x0}:y={bar_y}:w={max(2, x1 - x0)}:h={bar_h}:color={color}@0.45:t=fill')
        parts.append(f'drawbox=x={x0}:y={bar_y - 4}:w={max(2, x1 - x0)}:h={bar_h + 8}:color={color}@0.95:t=fill:enable=between(t\\,{s}\\,{e})')
        parts.append(f'drawbox=x={x0}:y={bar_y - 6}:w=2:h={bar_h + 12}:color=white:t=fill')
        label = f'SEG {i}/{n}   {s:.2f} - {e:.2f} s'
        parts.append(f"drawtext={font}text='{label}':x=12:y={12 + lane * (fs + 12)}:fontsize={fs}:fontcolor=white:box=1:boxcolor={color}@0.85:boxborderw=6:enable=between(t\\,{s}\\,{e})")
        parts.append(f'drawbox=x=0:y=0:w=iw:h=ih:color=red@0.9:t={max(6, h // 90)}:enable=between(t\\,{s}\\,{s + 0.35})')
    parts.append(f"drawtext={font}text='|':x=w*t/{duration}-{fs // 4}:y={bar_y - 10}:fontsize={bar_h + 20}:fontcolor=white:borderw=2:bordercolor=black")
    script = out / 'marked.filter'
    script.write_text(',\n'.join(parts))
    try:
        run(['ffmpeg', '-v', 'error', '-y', '-i', str(video), '-filter_script:v', str(script), '-map', '0:v:0', '-map', '0:a?', '-c:v', 'libx264', '-crf', '20', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-c:a', 'copy', str(out / 'marked.mp4')])
    finally:
        script.unlink(missing_ok=True)


def find_font(arg):
    for path in ([arg] if arg else FONTS):
        if path and Path(path).is_file():
            return "fontfile='" + path.replace('\\', '/').replace(':', '\\:') + "':"
    return ''


def main(argv=None):
    p = argparse.ArgumentParser(description='Preview Flora segments with ffmpeg: marked video and start/end frames.')
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--segments', type=Path, required=True, help='JSON with "segments": [{"start_s", "end_s"}].')
    p.add_argument('--out', type=Path, default=Path('preview'))
    p.add_argument('--font', help='TTF font file for the labels (auto-detected otherwise).')
    p.add_argument('--no-video', action='store_true')
    p.add_argument('--no-frames', action='store_true')
    args = p.parse_args(argv)
    try:
        for tool in ('ffmpeg', 'ffprobe'):
            if not shutil.which(tool):
                raise RuntimeError(f'{tool} not found in PATH.')
        w, h, fps, duration = probe(args.video)
        segments = load_segments(args.segments, duration)
        args.out.mkdir(parents=True, exist_ok=True)
        print(f'{args.video.name}: {w}x{h} @ {fps:.2f} fps, {duration:.2f} s, {len(segments)} segments')
        for i, (s, e) in enumerate(segments, 1):
            print(f'  {i:>2}  {s:8.2f} -> {e:8.2f}   ({e - s:.2f} s)')
        if not args.no_frames:
            paths = extract_frames(args.video, segments, fps, duration, args.out)
            contact_sheet(paths, args.out)
            print(f'frames: {args.out / "frames"}  contact sheet: {args.out / "contact_sheet.png"}')
        if not args.no_video:
            marked_video(args.video, segments, (w, h), duration, find_font(args.font), args.out)
            print(f'video: {args.out / "marked.mp4"}')
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f'segments_preview failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
