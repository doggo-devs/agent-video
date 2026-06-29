# vidgrid

Turn a video into VLM-readable grid images. Instead of feeding a model one frame
at a time, vidgrid samples frames, drops blurry and near-duplicate ones, scores
the rest, and packs them into grid images sized to a single pixel budget. Key
frames get larger cells so important moments keep their detail.

## Requirements

- Rust (edition 2024)
- `ffmpeg` on `PATH` (used for decoding and pre-scaling frames)

## Build

```bash
cargo build --release
# binary at target/release/vidgrid
```

## Usage

```bash
# Smart frame selection + grid composition
vidgrid process video.mp4 -o out/

# Target a specific model's pixel budget
vidgrid process video.mp4 --model gpt4o -o out/

# Override the pixel budget directly
vidgrid process video.mp4 --max-pixels 4096 -o out/

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

## Options

| Flag | Values | Default | Effect |
|------|--------|---------|--------|
| `--model` | `qwen2.5vl`, `gpt4o`, `claude`, `internvl` | `qwen2.5vl` | Seeds `--max-pixels` |
| `--max-pixels` | integer | from model | Canvas area budget |
| `--quality` | `low`, `medium`, `high` | `high` | Min frame size and canvas scale |
| `--detail` | `coarse`, `standard`, `fine` | `standard` | Sampling fps and dedup strictness |
| `--profile` | `balanced`, `fast`, `quality`, `action` | `balanced` | Named parameter preset |
| `--json` | flag | off | Output JSON with base64 to stdout |
| `-j` / `--threads` | integer | CPU cores | Rayon worker threads |
| `--in-flight` | integer | 2 | Max chunks buffered in pipeline |

Custom profiles can be defined in `~/.config/vidgrid/profiles.toml`:

```toml
[my-profile]
chunk_seconds = 45
jpeg_quality = 95
w_scene = 0.5
```

## How it works

1. **Extract** — ffmpeg samples frames at an fps set by `--detail`, streaming
   through a dedup filter that drops perceptual-hash duplicates.
2. **Score** — each frame gets a composite score from scene change, clarity,
   and edge density; top scorers become key frames.
3. **Lay out & compose** — frames are packed into a grid within the pixel budget,
   with key frames given larger cells, then rendered to JPEG.
