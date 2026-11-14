"""Run the path policy against real temporary Git trees, including mutations."""

from pathlib import Path
import subprocess
import sys
import tempfile


def check_policy(root: Path) -> None:
    """Verify rooted paths fail while configured paths and OS interfaces pass."""
    checker = Path(__file__).with_name("check_portable_paths.py")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    sample = root / "sample.cpp"
    sample.write_text('auto path = "input/sample.png";\n')
    subprocess.run(["git", "-C", str(root), "add", "sample.cpp"], check=True)
    command = [sys.executable, str(checker), str(root)]
    assert subprocess.run(command, capture_output=True).returncode == 0
    # Given arbitrary Unix and Windows deployment roots, including in comments.
    for literal in (
        "/" + "scratch/private/model.gguf",
        "/" + "mnt/data/model.gguf",
        "Z:" + "\\private\\model.gguf",
    ):
        # When the tracked file acquires a rooted literal, the actual CLI fails.
        sample.write_text(f'// deployment: {literal}\n')
        result = subprocess.run(command, text=True, capture_output=True)
        assert result.returncode == 1, result.stdout
        assert "sample.cpp:1:" in result.stdout, result.stdout
    # A literal prefix must not disguise another rooted storage literal.
    sample.write_text('auto path = "prefix" + "' + "/" + 'mnt/host/file";\n')
    assert subprocess.run(command, capture_output=True).returncode == 1
    sample.write_text("// portable input\n")
    link = root / "asset-link"
    link.symlink_to("/" + "scratch")
    subprocess.run(["git", "-C", str(root), "add", str(link)], check=True)
    assert subprocess.run(command, capture_output=True).returncode == 1
    link.unlink()
    link.symlink_to("input/sample.png")
    sample.write_text('auto path = "input/sample.png";\n')
    ignore = root / ".gitignore"
    ignore.write_text("# deployment: " + "/" + "mnt/host/file\n")
    subprocess.run(["git", "-C", str(root), "add", ".gitignore"], check=True)
    assert subprocess.run(command, capture_output=True).returncode == 1
    ignore.write_text("/" + "build/\n")
    # A changed or new file under a vendor directory is not automatically exempt.
    vendor = root / "thirdparty" / "stb" / "stb_image.h"
    vendor.parent.mkdir(parents=True)
    vendor.write_text("// deployment: " + "/" + "mnt/host/file\n")
    subprocess.run(["git", "-C", str(root), "add", str(vendor)], check=True)
    assert subprocess.run(command, capture_output=True).returncode == 1
    vendor.write_text("// portable replacement\n")
    # SDK exemptions must compare the whole Windows path, including spaces.
    sdk = root / "scripts" / "build-arm64.ps1"
    sdk.parent.mkdir()
    sdk.write_text("$path = 'C:" + "\\Program Files\\private\\model.gguf'\n")
    subprocess.run(["git", "-C", str(root), "add", str(sdk)], check=True)
    assert subprocess.run(command, capture_output=True).returncode == 1
    sdk.write_text("# configured SDK path\n")
    sample.write_text("#!" + "/" + "home/private/python\n")
    assert subprocess.run(command, capture_output=True).returncode == 1
    # Given configured prefixes, routes and a stable Linux interface.
    sample.write_text(
        'auto path = root + "/nested/file.png";\n'
        'auto route = "/api/inputs";\n'
        'auto identity = "' + "/" + 'proc/self/exe";\n'
    )
    assert subprocess.run(command, capture_output=True).returncode == 0
    print("PASS real Git path policy: relative/configured, Unix/Windows, comments, OS")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="trellis-path-policy-") as directory:
        check_policy(Path(directory))
    print("CLEANUP policy Git tree removed")
