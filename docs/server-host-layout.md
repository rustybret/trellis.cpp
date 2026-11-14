# Host web UI and storage layout

The host application adds an input library, persistent generation downloads and
an optional Blender turnaround endpoint to `trellis-server`. Generation still
uses the existing single-image API. No multiview inference is introduced.

The server reads these environment overrides once at startup. Unset or empty
values retain the exported host defaults:

| Variable | Default |
| --- | --- |
| `TRELLIS_INPUT_DIR` | `/mnt/gen4nvme/trellis.cpp/input` |
| `TRELLIS_OUTPUT_DIR` | `/mnt/gen4nvme/trellis.cpp/out` |
| `TRELLIS_WEB_DIR` | `web/` beside the binary's build directory, as in the host export |
| `TRELLIS_MODEL_VIEWER_JS` | `/mnt/gen4nvme/trellis.cpp/tools/mv_preview/model-viewer.min.js` |
| `TRELLIS_REVIEW_SCRIPT` | `/mnt/gen4nvme/trellis.cpp/tools/render_turnaround_8view.py` |
| `TRELLIS_REVIEW_LOG` | `/tmp/blender_review.log` |

Create writable input and output directories before use. The model directory
continues to use `--models`. A separate build directory must set `TRELLIS_WEB_DIR`
explicitly. Static JavaScript is served from the configured regular file through
`/model-viewer.min.js`; the host's absolute symlink is no longer needed. For the
same underlying file, the served JavaScript bytes are unchanged.

For an isolated installation, supply all paths explicitly:

```sh
export TRELLIS_INPUT_DIR="$PWD/input"
export TRELLIS_OUTPUT_DIR="$PWD/out"
export TRELLIS_WEB_DIR="$PWD/web"
export TRELLIS_MODEL_VIEWER_JS="$PWD/tools/mv_preview/model-viewer.min.js"
export TRELLIS_REVIEW_SCRIPT="$PWD/tools/render_turnaround_8view.py"
export TRELLIS_REVIEW_LOG="$PWD/review.log"
mkdir -p "$TRELLIS_INPUT_DIR" "$TRELLIS_OUTPUT_DIR"
./build/trellis-server --host 127.0.0.1 --port 8080 --models "$PWD/models"
```

`/api/inputs`, `/input-file/<name>`, upload archival, generated GLB/input-image
persistence, `/output-file/<name>` and `/api/review` share these configured
locations. Blender is an optional executable on `PATH`; review paths are
shell-quoted so spaces and apostrophes in installation paths are safe.

`tools/auto_preprocess.py` requires explicit input and output paths, supplied
as positional arguments or `TRELLIS_PREPROCESS_INPUT` and
`TRELLIS_PREPROCESS_OUTPUT`. Positional arguments take precedence. It has no
sample filename defaults. This utility change does not alter server defaults.

```sh
python3 tools/auto_preprocess.py input/subject.png input/processed.png
python3 tools/test_auto_preprocess.py
```

The utility and its regression require Pillow and NumPy.

## CPU-only verification

Configure with GPU backends disabled, then build the real application and the
default-contract test:

```sh
cmake -S . -B build-cpu -DGGML_METAL=OFF -DGGML_VULKAN=OFF \
  -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_BACKEND_DL=OFF \
  -DGGML_BLAS=OFF -DGGML_OPENMP=OFF -DTRELLIS_WEBP=OFF \
  -DTRELLIS_RETOPO_MATCHING=OFF -DTRELLIS_RETOPO_COLLISION=OFF
cmake --build build-cpu -j 2
./build-cpu/trellis-test-server-paths
python3 tools/test_server_portability.py "$PWD/build-cpu/trellis-server"
```

Run the default-contract executable with the listed server overrides unset.
The Python test launches the real CPU server on a temporary loopback port,
checks configured libraries, file bytes, static JavaScript and invalid-request
responses, and terminates its process. It never submits an image for inference
or invokes Blender. This does not establish inference fidelity, GPU memory use,
retopology-enabled Linux build compatibility or rendered review quality.
