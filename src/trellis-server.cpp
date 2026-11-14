// trellis-server — resident HTTP wrapper around the TRELLIS.2 image->3D pipeline.
//
//   GET  /health     -> "ok"
//   POST /generate    multipart/form-data with an "image" file part; optional text
//                      fields "seed", "resolution" (512/1024/1536), "bg_removal"
//                      (threshold|birefnet), "uv" (xatlas = default, unique
//                      chart space; box = faster projection), "band" (narrow-band
//                      DC remesh band width, default 1 — see --band), "model"
//                      (trellis|pixal3d — one directory holds both), "faces"
//                      (QEM decimation face target, >= 1000 or "auto" — see
//                      --faces), "atlas" (UV atlas px, 256..8192 or "auto"),
//                      "webp" (on|off; off = PNG textures), and for
//                      pixal3d the camera: "fov" (degrees), "mesh_scale",
//                      "extend_pixel", "texture" (on|off; off = geometry
//                      only, like --no-texture), and "normal_map" (on|off;
//                      on = bake a tangent-space normal map, like --normal-map).
//                      Returns model/gltf-binary.
//   GET  /api/inputs      -> JSON list of library input images in /mnt/gen4nvme/trellis.cpp/input/
//   GET  /input-file/:fn  -> Raw image file from input library
//   GET  /output-file/:fn -> Raw output file from /mnt/gen4nvme/trellis.cpp/out/
//   POST /api/review      -> Generates 8-view turnaround review panel (<model>.glb.png) in out/
//
#include "trellis_args.h"
#include "trellis_run.h"
#include "httplib.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include <stdexcept>

namespace {

std::string read_file_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

bool write_file_bytes(const std::string& path, const std::string& data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(data.data(), (std::streamsize) data.size());
    return f.good();
}

std::string temp_stem(const std::string& workdir = {}) {
    static std::atomic<unsigned> counter{0};
    std::error_code ec;
    std::filesystem::path dir = workdir.empty() ? std::filesystem::temp_directory_path(ec)
                                                : std::filesystem::absolute(workdir);
    if (ec) dir = ".";
    std::filesystem::create_directories(dir, ec);
    if (ec) throw std::runtime_error("cannot create request workdir: " + ec.message());
    auto n = counter.fetch_add(1);
    return (dir / ("trellis-req-" + std::to_string(n))).string();
}

std::string sanitize_filename(const std::string& raw) {
    size_t slash = raw.find_last_of("/\\");
    std::string base = (slash == std::string::npos) ? raw : raw.substr(slash + 1);
    std::string clean;
    for (char c : base) {
        if (isalnum((unsigned char)c) || c == '_' || c == '-' || c == '.') clean += c;
    }
    return clean.empty() ? "image.png" : clean;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    trellis::TrellisParams base;
    if (!trellis::parse_args(argc, argv, base)) {
        trellis::print_usage(argv[0], /*server=*/true);
        return base.help ? 0 : 1;
    }

    std::mutex gen_mu;
    httplib::Server svr;

    svr.set_post_routing_handler([](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        res.set_header("Access-Control-Expose-Headers", "X-Trellis-Saved-Glb");
        res.set_header("Access-Control-Max-Age", "86400");
    });
    svr.Options(R"(.*)", [](const httplib::Request&, httplib::Response& res) {
        res.status = 204;
    });

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("ok", "text/plain");
    });

    // Mount static web UI
    std::error_code web_ec;
    auto exe_p = std::filesystem::canonical(std::filesystem::path(argv[0]), web_ec);
    std::filesystem::path web_dir = exe_p.empty() ? std::filesystem::current_path() / "web"
                                                  : exe_p.parent_path().parent_path() / "web";
    if (std::filesystem::exists(web_dir, web_ec)) {
        svr.set_mount_point("/", web_dir.string());
        fprintf(stderr, "[trellis-server] web UI mounted from %s on /\n", web_dir.string().c_str());
    }

    // GET /api/inputs — list all available input images from /mnt/gen4nvme/trellis.cpp/input/
    svr.Get("/api/inputs", [](const httplib::Request&, httplib::Response& res) {
        std::string in_dir = "/mnt/gen4nvme/trellis.cpp/input";
        std::error_code ec;
        struct Item {
            std::filesystem::file_time_type t;
            std::string name;
            uintmax_t size;
        };
        std::vector<Item> items;
        for (const auto& entry : std::filesystem::directory_iterator(in_dir, ec)) {
            if (entry.is_regular_file(ec)) {
                auto ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".webp") {
                    items.push_back({entry.last_write_time(ec), entry.path().filename().string(), entry.file_size(ec)});
                }
            }
        }
        std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
            return a.t > b.t;
        });

        std::string json = "[";
        for (size_t i = 0; i < items.size(); ++i) {
            if (i > 0) json += ",";
            json += "{\"name\":\"" + items[i].name + "\",\"size\":" + std::to_string(items[i].size) + ",\"url\":\"/input-file/" + items[i].name + "\"}";
        }
        json += "]";
        res.set_content(json, "application/json");
    });

    // GET /input-file/:name — serve raw image from input library
    svr.Get(R"(/input-file/(.+))", [](const httplib::Request& req, httplib::Response& res) {
        std::string name = sanitize_filename(req.matches[1]);
        std::string path = "/mnt/gen4nvme/trellis.cpp/input/" + name;
        std::string bytes = read_file_bytes(path);
        if (bytes.empty()) {
            res.status = 404;
            res.set_content("{\"error\":\"input file not found\"}", "application/json");
            return;
        }
        std::string ext = std::filesystem::path(name).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        std::string mime = (ext == ".jpg" || ext == ".jpeg") ? "image/jpeg"
                         : (ext == ".webp") ? "image/webp" : "image/png";
        res.set_content(bytes, mime.c_str());
    });

    // GET /output-file/:name — serve file from output directory
    svr.Get(R"(/output-file/(.+))", [](const httplib::Request& req, httplib::Response& res) {
        std::string name = sanitize_filename(req.matches[1]);
        std::string path = "/mnt/gen4nvme/trellis.cpp/out/" + name;
        std::string bytes = read_file_bytes(path);
        if (bytes.empty()) {
            res.status = 404;
            res.set_content("{\"error\":\"output file not found\"}", "application/json");
            return;
        }
        std::string ext = std::filesystem::path(name).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        std::string mime = (ext == ".png") ? "image/png"
                         : (ext == ".glb") ? "model/gltf-binary"
                         : (ext == ".obj" || ext == ".ply") ? "text/plain" : "application/octet-stream";
        res.set_content(bytes, mime.c_str());
    });

    // POST /api/review — trigger 8-view turnaround sheet generator for a generated GLB
    svr.Post("/api/review", [](const httplib::Request& req, httplib::Response& res) {
        std::string glb_name;
        if (req.has_param("glb")) {
            glb_name = req.get_param_value("glb");
        } else {
            // Check JSON body {"glb": "..."}
            auto pos = req.body.find("\"glb\"");
            if (pos != std::string::npos) {
                auto colon = req.body.find(':', pos);
                auto q1 = req.body.find('"', colon + 1);
                auto q2 = req.body.find('"', q1 + 1);
                if (q1 != std::string::npos && q2 != std::string::npos) {
                    glb_name = req.body.substr(q1 + 1, q2 - q1 - 1);
                }
            }
        }
        if (glb_name.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"missing 'glb' parameter\"}", "application/json");
            return;
        }

        std::string clean_glb = sanitize_filename(glb_name);
        std::string out_dir = "/mnt/gen4nvme/trellis.cpp/out";
        std::string glb_path = out_dir + "/" + clean_glb;
        std::string png_name = clean_glb + ".png";
        std::string png_path = out_dir + "/" + png_name;

        if (!std::filesystem::exists(glb_path)) {
            res.status = 404;
            res.set_content("{\"error\":\"glb file not found: " + clean_glb + "\"}", "application/json");
            return;
        }

        std::string cmd = "blender --background --python /mnt/gen4nvme/trellis.cpp/tools/render_turnaround_8view.py -- "
                        + glb_path + " " + png_path + " > /tmp/blender_review.log 2>&1";
        int rc = std::system(cmd.c_str());
        if (rc != 0 || !std::filesystem::exists(png_path)) {
            res.status = 500;
            res.set_content("{\"error\":\"8-view render failed with code " + std::to_string(rc) + "\"}", "application/json");
            return;
        }

        res.set_content("{\"status\":\"ok\",\"glb\":\"" + clean_glb + "\",\"review_png\":\"" + png_name + "\",\"review_url\":\"/output-file/" + png_name + "\"}", "application/json");
    });

    // POST /generate — full 3D reconstruction pipeline
    svr.Post("/generate", [&](const httplib::Request& req, httplib::Response& res) {
        if (!req.has_file("image")) {
            res.status = 400;
            res.set_content("{\"error\":\"missing 'image' file part\"}", "application/json");
            return;
        }
        const auto& image = req.get_file_value("image");

        trellis::TrellisParams p = base;
        if (req.has_file("seed")) p.seed = (uint32_t) atoi(req.get_file_value("seed").content.c_str());
        if (req.has_file("resolution")) p.set_res(atoi(req.get_file_value("resolution").content.c_str()));
        if (req.has_file("bg_removal")) p.birefnet = (req.get_file_value("bg_removal").content == "birefnet") ? 1 : 0;
        if (req.has_file("uv")) p.xatlas = (req.get_file_value("uv").content == "xatlas");
        if (req.has_file("band")) p.band = atoi(req.get_file_value("band").content.c_str());
        if (req.has_file("atlas")) {
            std::string error;
            if (!trellis::parse_atlas_arg(req.get_file_value("atlas").content.c_str(), p, error)) {
                res.status = 400;
                res.set_content(error + "\n", "text/plain");
                return;
            }
        }
        if (req.has_file("steps")) p.steps = atoi(req.get_file_value("steps").content.c_str());
        if (req.has_file("tex_res")) p.tex_res = atoi(req.get_file_value("tex_res").content.c_str());
        if (req.has_file("fa_fast")) p.fa_fast = (req.get_file_value("fa_fast").content == "0" || req.get_file_value("fa_fast").content == "false") ? 0 : 1;
        if (req.has_file("faces")) {
            std::string error;
            if (!trellis::parse_faces_arg(req.get_file_value("faces").content.c_str(), p, error) ||
                !trellis::check_faces_compat(p, error)) {
                res.status = 400;
                res.set_content(error + "\n", "text/plain");
                return;
            }
        }
        if (req.has_file("model")) {
            const std::string& m = req.get_file_value("model").content;
            if      (m == "pixal3d") p.family = trellis::ModelFamily::Pixal3D;
            else if (m == "trellis") p.family = trellis::ModelFamily::Trellis;
            else { res.status = 400; res.set_content("model must be trellis or pixal3d\n", "text/plain"); return; }
        }
        for (const auto& field : {std::pair<const char*, const char*>{"fov", "fov"},
                                  {"mesh_scale", "mesh-scale"}, {"extend_pixel", "extend-pixel"}}) {
            if (!req.has_file(field.first)) continue;
            std::string error;
            if (!trellis::parse_camera_arg(field.second, req.get_file_value(field.first).content.c_str(), p, error)) {
                res.status = 400;
                res.set_content(error + "\n", "text/plain");
                return;
            }
        }
        if (req.has_file("texture")) {
            const std::string& t = req.get_file_value("texture").content;
            if      (t == "off" || t == "0" || t == "false") p.texture = false;
            else if (t == "on"  || t == "1" || t == "true")  p.texture = true;
            else { res.status = 400; res.set_content("texture must be on or off\n", "text/plain"); return; }
        }
        if (req.has_file("normal_map")) {
            const std::string& n = req.get_file_value("normal_map").content;
            if      (n == "off" || n == "0" || n == "false") p.normal_map = 0;
            else if (n == "on"  || n == "1" || n == "true")  p.normal_map = 1;
            else { res.status = 400; res.set_content("normal_map must be on or off\n", "text/plain"); return; }
        }
        if (req.has_file("webp")) {
            const std::string& w = req.get_file_value("webp").content;
            p.webp = (w == "off" || w == "0" || w == "false") ? 0
                   : (w == "on"  || w == "1" || w == "true")  ? 1 : -1;
        }

        // Save submitted image into input library (/mnt/gen4nvme/trellis.cpp/input/)
        {
            std::string in_dir = "/mnt/gen4nvme/trellis.cpp/input";
            std::time_t t_now = std::time(nullptr);
            char in_ts[64];
            std::strftime(in_ts, sizeof(in_ts), "%Y%m%d_%H%M%S", std::localtime(&t_now));
            std::string in_name;
            if (!image.filename.empty()) {
                std::string base_fn = sanitize_filename(image.filename);
                in_name = std::string(in_ts) + "_" + base_fn;
            } else {
                in_name = "upload_" + std::string(in_ts) + "_s" + std::to_string(p.seed) + ".png";
            }
            write_file_bytes(in_dir + "/" + in_name, image.content);
            fprintf(stderr, "[trellis-server] archived input to %s/%s\n", in_dir.c_str(), in_name.c_str());
        }

        std::string stem;
        try {
            stem = temp_stem(p.retopo ? p.retopo_workdir : std::string{});
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content("{\"error\":\"failed to stage request files\"}", "application/json");
            return;
        }
        p.image  = stem + ".png";
        p.output = stem + ".glb";

        std::string glb;
        std::string saved_glb_base;
        std::string error_message = "3D reconstruction failed";
        {
            std::lock_guard<std::mutex> lk(gen_mu);
            if (!write_file_bytes(p.image, image.content)) {
                res.status = 500;
                res.set_content("{\"error\":\"failed to stage input image\"}", "application/json");
                return;
            }
            fprintf(stderr, "[trellis-server] generate: %zu-byte image, seed %u, res %s, bg %s, uv %s, faces %d, atlas %s, webp %s, texture %s\n",
                    image.content.size(), p.seed, p.cascade ? std::to_string(p.hr_res).c_str() : "512",
                    p.birefnet < 0 ? "auto" : (p.birefnet ? "birefnet" : "threshold"), p.xatlas ? "xatlas" : "box",
                    trellis::qem_face_target(p), p.tex > 0 ? std::to_string(p.tex).c_str() : "auto",
                    p.webp < 0 ? "auto" : (p.webp ? "on" : "off"), p.texture ? "on" : "off");
            try {
                int rc = trellis_run(p);
                if (rc == 0) {
                    glb = read_file_bytes(p.output);
                    try {
                        std::time_t t = std::time(nullptr);
                        char ts[64];
                        std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", std::localtime(&t));
                        std::string out_dir = "/mnt/gen4nvme/trellis.cpp/out";
                        std::string model_prefix = (p.family == trellis::ModelFamily::Pixal3D ? "pixal3d" : "trellis");
                        saved_glb_base = model_prefix + "_" + ts + "_s" + std::to_string(p.seed) + ".glb";
                        std::string saved_glb = out_dir + "/" + saved_glb_base;
                        std::string saved_png = out_dir + "/" + model_prefix + "_" + ts + "_s" + std::to_string(p.seed) + "_input.png";
                        write_file_bytes(saved_glb, glb);
                        write_file_bytes(saved_png, image.content);
                        fprintf(stderr, "[trellis-server] persisted generation to %s\n", saved_glb.c_str());
                    } catch (const std::exception& save_err) {
                        fprintf(stderr, "[trellis-server] warning: failed to persist to out/: %s\n", save_err.what());
                    }
                }
            } catch (const std::exception& e) {
                fprintf(stderr, "[trellis-server] generate failed: %s\n", e.what());
                error_message = e.what();
            }
            std::remove(p.image.c_str());
            std::remove(p.output.c_str());
            std::remove((stem + ".ply").c_str());
            std::remove((stem + "_base.png").c_str());
        }

        if (glb.empty()) {
            res.status = 500;
            std::string escaped;
            for (char c : error_message) {
                switch (c) {
                    case '"':  escaped += "\\\""; break;
                    case '\\': escaped += "\\\\"; break;
                    case '\n': escaped += "\\n";  break;
                    case '\r': escaped += "\\r";  break;
                    case '\t': escaped += "\\t";  break;
                    default:
                        if ((unsigned char)c < 0x20) break;
                        escaped += c;
                }
            }
            res.set_content("{\"error\":\"" + escaped + "\"}", "application/json");
            return;
        }

        if (!saved_glb_base.empty()) {
            res.set_header("X-Trellis-Saved-Glb", saved_glb_base);
        }
        res.set_content(glb.data(), glb.size(), "model/gltf-binary");
    });

    fprintf(stderr, "[trellis-server] models=%s gpu=%d listening on http://%s:%d\n",
            base.models.c_str(), base.gpu, base.host.c_str(), base.port);
    if (!svr.listen(base.host, base.port)) {
        fprintf(stderr, "[trellis-server] failed to bind %s:%d\n", base.host.c_str(), base.port);
        return 1;
    }
    return 0;
}
