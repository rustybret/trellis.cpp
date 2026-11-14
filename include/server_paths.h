#pragma once

#include <cstdlib>
#include <string>

namespace trellis {

inline std::string server_path(const char* variable, const std::string& fallback) {
    const char* value = std::getenv(variable);
    return value && *value ? value : fallback;
}

// Read once at server startup. Defaults are relative to the working directory.
struct ServerPaths {
    std::string input = server_path("TRELLIS_INPUT_DIR", "input");
    std::string output = server_path("TRELLIS_OUTPUT_DIR", "out");
    std::string viewer = server_path("TRELLIS_MODEL_VIEWER_JS",
        "tools/mv_preview/model-viewer.min.js");
    std::string review_script = server_path("TRELLIS_REVIEW_SCRIPT",
        "tools/render_turnaround_8view.py");
    std::string review_log = server_path("TRELLIS_REVIEW_LOG", "review.log");
};

} // namespace trellis
