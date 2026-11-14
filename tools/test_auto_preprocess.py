"""Run the actual preprocessing CLI with explicit, neutral test files."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile

from PIL import Image


def check_cli() -> None:
    script = Path(__file__).with_name("auto_preprocess.py")
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith("TRELLIS_PREPROCESS_")
    }
    with tempfile.TemporaryDirectory(prefix="preprocess ") as temporary:
        directory = Path(temporary)
        source = directory / "subject.png"
        output = directory / "processed.png"
        image = Image.new("RGBA", (32, 64), (0, 0, 0, 0))
        image.paste((120, 80, 40, 255), (8, 8, 24, 56))
        image.save(source)

        # Given no explicit file selection, fail before processing any sample.
        missing = subprocess.run(
            [sys.executable, str(script)], env=environment,
            capture_output=True, text=True, check=False,
        )
        assert missing.returncode == 2, (missing.returncode, missing.stderr)
        assert "input" in missing.stderr and "output" in missing.stderr

        # When both paths are explicit, exercise the real Pillow/NumPy utility.
        processed = subprocess.run(
            [sys.executable, str(script), str(source), str(output)],
            env=environment, capture_output=True, text=True, check=False,
        )
        assert processed.returncode == 0, processed.stderr
        with Image.open(output) as result:
            assert result.size == (1024, 1024)
            assert result.mode == "RGBA"
            assert result.getchannel("A").getbbox() is not None
        print("PASS missing paths rejected; explicit CLI paths produce nonempty RGBA")


if __name__ == "__main__":
    check_cli()
