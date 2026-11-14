#pragma once

#include <cstdlib>
#include <string>

namespace trellis {

inline std::string server_path(const char* variable, const std::string& fallback) {
    const char* value = std::getenv(variable);
    return value && *value ? value : fallback;
}

// Read once at server startup. Unset or empty overrides preserve the host layout.
struct ServerPaths {
    std::string input = server_path("TRELLIS_INPUT_DIR", "/mnt/gen4nvme/trellis.cpp/input");
    std::string output = server_path("TRELLIS_OUTPUT_DIR", "/mnt/gen4nvme/trellis.cpp/out");
    std::string viewer = server_path("TRELLIS_MODEL_VIEWER_JS",
        "/mnt/gen4nvme/trellis.cpp/tools/mv_preview/model-viewer.min.js");
    std::string review_script = server_path("TRELLIS_REVIEW_SCRIPT",
        "/mnt/gen4nvme/trellis.cpp/tools/render_turnaround_8view.py");
    std::string review_log = server_path("TRELLIS_REVIEW_LOG", "/tmp/blender_review.log");
};

} // namespace trellis
