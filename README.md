# vidgrid

Turn a video into VLM-readable grid images. Instead of feeding a model one frame
at a time, vidgrid samples frames, drops blurry and near-duplicate ones, scores
the rest, and packs them into grid images sized to a single pixel budget. Key
frames get larger cells so important moments keep their detail.

## Requirements

- Rust (edition 2024)
- `ffmpeg` on `PATH` (decoding and pre-scaling frames)
- `ffprobe` on `PATH` (ships with ffmpeg; verifies duration in the default
  fixed-fps mode and probes keyframes for `--strategy keyframes`/`adaptive`.
  `--strategy fixed-fps --force` and `extract` run without it, skipping
  embedded-subtitle auto-detection with a warning)

## Build

```bash
cargo build --release
# binary at target/release/vidgrid
```

`.cargo/config.toml` targets `x86-64-v3` (AVX2, ~all x86-64 CPUs since 2015).
When distributing to older hardware, override with
`RUSTFLAGS="-C target-cpu=x86-64"`.

## Usage

```bash
# Smart frame selection + grid composition
vidgrid process video.mp4 -o out/

# Override the canvas edge length
vidgrid process video.mp4 --max-edge 4096 -o out/

# High-density sampling of one time range into a single grid
vidgrid extract video.mp4 --range 01:23-01:35 -o out/

# Output JSON with base64 images to stdout (no files written)
vidgrid process video.mp4 -o /dev/null --json

# Use a named profile
vidgrid process video.mp4 --profile fast -o out/

# Control parallelism
vidgrid process video.mp4 -j 4 --in-flight 2 -o out/
```

Output is written to the directory given by `-o`: one or more `grid_*.jpg` images
plus a `metadata.json` describing each grid, its time range, and its key frames.

File output streams grids to disk as they finish, so memory stays flat over any
video length. `--json` instead holds every grid in memory (base64-inline) until
the run completes — fine for inline consumption of short videos, but prefer
file output for long ones.

## Options

| Flag | Values | Default | Effect |
|------|--------|---------|--------|
| `--max-edge` | integer | `2048` | Canvas edge length in pixels |
| `--quality` | `low`, `medium`, `high` | `low` | Cell size floor; lower packs more frames per grid |
| `--detail` | `coarse`, `standard`, `fine` | `standard` | Sampling fps |
| `--profile` | `balanced`, `fast`, `quality`, `action` | `balanced` | Named parameter preset |
| `--strategy` | `fixed-fps`, `keyframes`, `adaptive` | `fixed-fps` | Frame sampling strategy |
| `--json` | flag | off | Output JSON with base64 to stdout |
| `-j` / `--threads` | integer | CPU cores | Rayon worker threads |
| `--in-flight` | integer | 2 | Max chunks buffered in pipeline |

Run `vidgrid process --help` for the full list of override knobs (sampling,
scoring weights, timing, subtitles).

## How it works

1. **Extract** — ffmpeg samples frames at an fps set by `--detail`, streaming
   through a dedup filter that drops perceptual-hash duplicates.
2. **Score** — each frame gets a composite score from scene change, clarity,
   and edge density; top scorers become key frames.
3. **Lay out & compose** — frames are packed into a grid within the pixel budget,
   with key frames given larger cells, then rendered to JPEG.
