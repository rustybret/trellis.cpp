"""Exercise the real CPU server without loading models or running inference.

Run: python3 tools/test_server_portability.py build-cpu/trellis-server
"""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


def check_server(binary: Path, directory: Path) -> None:
    """Verify custom libraries, static assets and validation over real HTTP."""
    # Given an isolated installation whose paths contain shell metacharacters.
    input_dir = directory / "input library"
    output_dir = directory / "output library"
    web_dir = directory / "web"
    for path in (input_dir, output_dir, web_dir):
        path.mkdir()
    image = b"\x89PNG\r\n\x1a\nportable-input"
    (input_dir / "sample.png").write_bytes(image)
    (output_dir / "sample.glb").write_bytes(b"portable-output")
    (web_dir / "index.html").write_text("<html><body>portable-ui</body></html>")
    viewer = directory / "viewer.js"
    viewer.write_text("/* portable-viewer */")
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith("TRELLIS_")
    }
    environment.update(
        TRELLIS_INPUT_DIR=str(input_dir),
        TRELLIS_OUTPUT_DIR=str(output_dir),
        TRELLIS_WEB_DIR=str(web_dir),
        TRELLIS_MODEL_VIEWER_JS=str(viewer),
    )
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    with (directory / "server.log").open("w+") as log:
        process = subprocess.Popen(
            [str(binary), "--host", "127.0.0.1", "--port", str(port),
             "--gpu", "-1", "--backend", "CPU"],
            env=environment, stdout=log, stderr=log,
        )
        try:
            deadline = time.monotonic() + 15
            while True:
                try:
                    with urllib.request.urlopen(url + "/health", timeout=1) as response:
                        assert response.read() == b"ok"
                    break
                except urllib.error.URLError:
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise

            # When requesting the actual configured libraries and static routes.
            with urllib.request.urlopen(url + "/api/inputs") as response:
                library = json.load(response)
            assert library == [
                {"name": "sample.png", "size": len(image),
                 "url": "/input-file/sample.png"}
            ], library
            for route, expected in (
                ("/", b"<html><body>portable-ui</body></html>"),
                ("/input-file/sample.png", image),
                ("/output-file/sample.glb", b"portable-output"),
                ("/model-viewer.min.js", b"/* portable-viewer */"),
            ):
                with urllib.request.urlopen(url + route) as response:
                    assert response.status == 200
                    assert response.read() == expected, route

            # Then invalid requests retain the upstream validation contracts.
            for route, body, status in (
                ("/api/review", b"{}", 400),
                ("/api/review", b'{"glb":"missing.glb"}', 404),
                ("/generate", b"", 400),
            ):
                request = urllib.request.Request(
                    url + route, data=body,
                    headers={"Content-Type": "application/json"},
                )
                try:
                    urllib.request.urlopen(request)
                except urllib.error.HTTPError as error:
                    assert error.code == status, (route, error.code)
                else:
                    raise AssertionError(f"{route} accepted invalid input")
            print("PASS real HTTP custom paths, static viewer, empty/missing validation")
        finally:
            process.terminate()
            process.wait(timeout=10)
            assert process.poll() is not None
            print(f"CLEANUP server pid={process.pid} exited={process.returncode}")


if __name__ == "__main__":
    executable = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="trellis paths ' $; ") as temporary:
        check_server(executable, Path(temporary))
