#include "server_paths.h"

#include <cstdio>
#include <cstdlib>

int main() {
    // Given no deployment overrides, paths stay relative to the working directory.
    for (const char* key : {"TRELLIS_INPUT_DIR", "TRELLIS_OUTPUT_DIR",
                            "TRELLIS_MODEL_VIEWER_JS", "TRELLIS_REVIEW_SCRIPT",
                            "TRELLIS_REVIEW_LOG", "TRELLIS_WEB_DIR"}) {
        if (std::getenv(key)) {
            std::fprintf(stderr, "run this default-contract test with %s unset\n", key);
            return 1;
        }
    }
    const trellis::ServerPaths paths;
    if (paths.input != "input" ||
        paths.output != "out" ||
        paths.viewer != "tools/mv_preview/model-viewer.min.js" ||
        paths.review_script != "tools/render_turnaround_8view.py" ||
        paths.review_log != "review.log" ||
        trellis::server_path("TRELLIS_WEB_DIR", "web") != "web") {
        std::fprintf(stderr, "server path defaults must be portable\n");
        return 1;
    }
    std::puts("PASS portable server path defaults");
    return 0;
}
