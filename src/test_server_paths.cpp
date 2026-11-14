#include "server_paths.h"

#include <cstdio>
#include <cstdlib>

int main() {
    // Given no deployment overrides, the exported production locations survive.
    for (const char* key : {"TRELLIS_INPUT_DIR", "TRELLIS_OUTPUT_DIR",
                            "TRELLIS_MODEL_VIEWER_JS", "TRELLIS_REVIEW_SCRIPT",
                            "TRELLIS_REVIEW_LOG", "TRELLIS_WEB_DIR"}) {
        if (std::getenv(key)) {
            std::fprintf(stderr, "run this default-contract test with %s unset\n", key);
            return 1;
        }
    }
    const trellis::ServerPaths paths;
    if (paths.input != "/mnt/gen4nvme/trellis.cpp/input" ||
        paths.output != "/mnt/gen4nvme/trellis.cpp/out" ||
        paths.viewer != "/mnt/gen4nvme/trellis.cpp/tools/mv_preview/model-viewer.min.js" ||
        paths.review_script != "/mnt/gen4nvme/trellis.cpp/tools/render_turnaround_8view.py" ||
        paths.review_log != "/tmp/blender_review.log" ||
        trellis::server_path("TRELLIS_WEB_DIR", "/installation/web") != "/installation/web") {
        std::fprintf(stderr, "production path defaults changed\n");
        return 1;
    }
    std::puts("PASS unchanged production path defaults");
    return 0;
}
