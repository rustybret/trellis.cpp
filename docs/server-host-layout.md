# Host web UI and storage layout

The host application adds an input library, persistent generation downloads and
an optional Blender turnaround endpoint to `trellis-server`. Generation still
uses the existing single-image API. No multiview inference is introduced.

The server reads these environment overrides once at startup. Unset or empty
values use these defaults, resolved relative to the server's working directory:

| Variable | Default |
| --- | --- |
| `TRELLIS_INPUT_DIR` | `input` |
| `TRELLIS_OUTPUT_DIR` | `out` |
| `TRELLIS_WEB_DIR` | `web` |
| `TRELLIS_MODEL_VIEWER_JS` | `tools/mv_preview/model-viewer.min.js` |
| `TRELLIS_REVIEW_SCRIPT` | `tools/render_turnaround_8view.py` |
| `TRELLIS_REVIEW_LOG` | `review.log` |

Create writable input and output directories before use. The model directory
continues to use `--models`. Start the server from the checkout root, or set `TRELLIS_WEB_DIR`
and the other paths explicitly. Static JavaScript is served from the configured regular file through
`/model-viewer.min.js`; no host symlink is needed. For the
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
sample filename defaults.

```sh
python3 tools/auto_preprocess.py input/subject.png input/processed.png
python3 tools/test_auto_preprocess.py
```

The utility and its regression require Pillow and NumPy.

## Path policy

Building requires Python 3 and Git. The core build runs
`tools/check_portable_paths.py` against tracked first-party text, including
comments and documentation. Deployment, storage, model and reference-checkout
locations must be configured or relative. Run its real Git-tree regression with
`python3 tools/test_portable_paths.py`.

The check lists stable operating-system interfaces and inherited SDK defaults
explicitly. Vendored files are exempt only while their Git content identifiers
match the pinned upstream versions; a modified vendor file is scanned. The
existing ggml submodule revision is listed separately. URLs, HTTP routes,
interpreter directives and repository-relative Git ignore patterns are syntax,
not deployment storage settings.

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
