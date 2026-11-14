"""Reject rooted storage paths in first-party tracked text, including comments.

Unmodified upstream dependencies are excluded by exact Git content identifier.
URLs, HTTP routes, MIME types, shebang interpreter directives and anchored Git
ignore patterns are syntax rather than deployment storage locations.
"""

from pathlib import Path
import hashlib
import re
import subprocess
import sys
from typing import Final


# Unchanged vendored sources at upstream 6851891. Content IDs work in shallow
# clones too; modifying a vendor file removes its exemption automatically.
UPSTREAM: Final = {
    "app/public/vendor/model-viewer.min.js": "ee8c68547725335444be884201cb39478b530092",
    "tools/mv_preview/model-viewer.min.js": "ee8c68547725335444be884201cb39478b530092",
    "thirdparty/cpp-httplib/httplib.h": "eb592d2e7a72593915e20632ba94eeb2076e4a40",
    "thirdparty/fqms/LICENSE.md": "489accfd83ae148505d94fd1981d2f4f5a30fe66",
    "thirdparty/fqms/Simplify.h": "674e192ea4a93980bc52b2902d8a212231d28427",
    "thirdparty/meshoptimizer/LICENSE.md": "a5c3b1ccca7bbe958d17644bde2ec1fc6f14c19d",
    "thirdparty/meshoptimizer/meshoptimizer.h": "eacd14df28a2b85c58bd846fb439225581d81762",
    "thirdparty/meshoptimizer/simplifier.cpp": "1abcd51f54574344f62e47f4bee3d39b52daae61",
    "thirdparty/stb/stb_image.h": "5e807a0a6e7cdbfbbf48dff5f5d3f3693c2bc851",
    "thirdparty/stb/stb_image_resize.h": "bcca92c2dca11a31355e3f026e8779d2d522d33c",
    "thirdparty/stb/stb_image_write.h": "9128a3131f8ce0c9571078bd374c6a4aa808d4f3",
    "thirdparty/xatlas/xatlas.cpp": "5c5c57ecabbdf5fa8f95c7c5890b5b55b2911c9a",
    "thirdparty/xatlas/xatlas.h": "d66a96db21dee5f326ab4d60eb20eb8b1aac7e2d",
}
GGML_COMMIT: Final = "f8697b33795db2ee6a2569b40853ca6aaa69f9b9"
PATH: Final = re.compile(
    r"(?<![\w./:\\}~%>])(?:[A-Za-z]:[\\/][^\r\n\"'`<>|;)]*"
    r"|/(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.*${}-]*"
    r"|(?<=[\"'])/[A-Za-z0-9_.-]+(?=[\"'])"
    r"|/(?:mnt|home|Users|Volumes|tmp|usr|opt|var|etc|media|run|root|srv|Applications|private)\b)"
)
EXCLUSIONS: Final = {
    # Stable operating-system interfaces, not deployment storage.
    "/dev/null": "OS null device",
    "/usr/bin/env": "POSIX interpreter lookup for the listed shebang directives",
    "/proc/self/exe": "Linux executable identity",
    "/proc/self/statm": "Linux process memory statistics",
    "/sys/class/drm/card1/device/mem_info_gtt_used": "Linux GPU memory statistics",
}
TOOLCHAIN: Final = {
    # Upstream toolchain defaults retained under the lead's explicit scope ruling.
    "CMakeLists.txt": ("/usr/local/cuda-13.3/bin/nvcc", "/opt/rocm"),
    ".github/workflows/release.yml": (
        "/etc/apt/apt-mirrors.txt", "/etc/apt/sources.list",
        "/etc/apt/sources.list.d/ubuntu.sources",
        "/etc/apt/apt.conf.d/99-trellis-network", "/opt/rocm",
        "/opt/rocm/lib", "/opt/rocm/lib64", "/opt/rocm/llvm/lib",
        "/opt/rocm/llvm/bin/clang", "/opt/rocm/llvm/bin/clang++",
        "/opt/rocm/bin", "/opt/rocm/llvm/bin",
        "/usr/local/cuda",
    ),
    "scripts/build-arm64.ps1": (
        "C:\\Qualcomm\\Hexagon_SDK\\6.4.0.0", "C:\\VulkanSDK",
        "C:\\Program Files\\LLVM\\bin\\clang.exe",
    ),
}
SYNTAX: Final = {
    "install/install.ps1": ("/S",),  # Windows installer silent flag.
    "src/trellis_cli.cpp": ("/pixal3d_",),  # Filename suffix joined to model root.
}
WEB: Final = re.compile(
    r"https?://[^\s\"'`<>]+"
    r"|/(?:api|input-file|output-file)(?:/[A-Za-z0-9_.:<>()?+%-]*)*"
    r"|/(?:generate|health|model-viewer\.min\.js)\b"
    r"|/(?:shot\.png|models\.json|view\.html)\b"
    r"|(?:application|model|multipart|text|image)/[A-Za-z0-9.+-]+"
)


def audit(root: Path) -> list[str]:
    """Return every storage-path violation in the Git tracked tree."""
    names = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z"], text=True,
    ).split("\0")
    findings: list[str] = []
    for name in names:
        if not name:
            continue
        path = root / name
        if path.is_symlink():
            target = path.readlink()
            if target.is_absolute() or re.match(r"^[A-Za-z]:[\\/]", str(target)):
                findings.append(f"{name}: absolute symlink target {target}")
            continue
        if path.is_dir():
            entry = subprocess.check_output(
                ["git", "-C", str(root), "ls-files", "--stage", "--", name], text=True,
            )
            if name != "thirdparty/ggml" or GGML_COMMIT not in entry:
                findings.append(f"{name}: unlisted dependency revision")
            continue
        data = path.read_bytes()
        blob = hashlib.sha1(
            f"blob {len(data)}\0".encode() + data, usedforsecurity=False,
        ).hexdigest()
        if UPSTREAM.get(name) == blob:
            continue
        if b"\0" in data:
            continue
        text = data.decode("utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), 1):
            if line in (
                "#!/usr/bin/env python3", "#!/usr/bin/env python", "#!/usr/bin/env bash",
            ):
                continue
            if name == ".gitignore" and not line.lstrip().startswith("#"):
                continue  # Git patterns are relative to the repository, including anchors.
            for match in PATH.finditer(WEB.sub("", line)):
                value = match.group().rstrip(".}")
                if (
                    value in EXCLUSIONS or value in TOOLCHAIN.get(name, ())
                    or value in SYNTAX.get(name, ())
                ):
                    continue
                if name == "tools/check_portable_paths.py" and (
                    value.replace("\\\\", "\\") in {
                        item for paths in (*TOOLCHAIN.values(), *SYNTAX.values())
                        for item in paths
                    }
                ):
                    continue
                # A suffix after interpolation or string concatenation is relative
                # to that configured prefix, not a rooted filesystem literal.
                prefix = WEB.sub("", line)[:match.start()]
                if re.search(r'(?:\+|<<)\s*["\']$', prefix) and not re.search(
                    r'["\']\s*(?:\+|<<)\s*["\']$', prefix,
                ):
                    continue
                findings.append(f"{name}:{number}: {match.group()}")
    return findings


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    violations = audit(root)
    print("\n".join(violations) if violations else "PASS portable first-party paths")
    sys.exit(bool(violations))
