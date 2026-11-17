#include "multiview.h"
#include "trellis_args.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

bool run_args(const std::vector<std::string>& words, trellis::TrellisParams& p) {
    std::vector<char*> argv;
    for (const auto& word : words) argv.push_back(const_cast<char*>(word.c_str()));
    return trellis::parse_args(static_cast<int>(argv.size()), argv.data(), p);
}

int fail(const char* test, const std::string& detail) {
    std::fprintf(stderr, "FAIL %s: %s\n", test, detail.c_str());
    return 1;
}

int expect_reject(const char* test, const std::vector<std::string>& words) {
    trellis::TrellisParams p;
    if (run_args(words, p)) return fail(test, "parse unexpectedly succeeded");
    std::printf("PASS %s\n", test);
    return 0;
}

}

int main(int argc, char** argv) {
    std::string which = "all";
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--case" && i + 1 < argc) which = argv[++i];
        else which = argv[i];
    }
    int rc = 0;

    if (which == "all" || which == "legacy") {
        trellis::TrellisParams p;
        if (!run_args({"trellis-cli", "front.png", "model.glb"}, p)) rc |= fail("legacy-positional", "parse failed");
        else if (p.image != "front.png" || p.output != "model.glb" || p.multiview.active()) rc |= fail("legacy-positional", "unexpected params");
        else std::printf("PASS legacy-positional\n");

        trellis::TrellisParams q;
        if (!run_args({"trellis-cli", "--image", "front.png", "--output", "model.glb"}, q)) rc |= fail("legacy-flag", "parse failed");
        else if (q.image != "front.png" || q.output != "model.glb" || q.multiview.active()) rc |= fail("legacy-flag", "unexpected params");
        else std::printf("PASS legacy-flag\n");
        trellis::TrellisParams r;
        if (!run_args({"trellis-cli", "front.png", "early.glb", "--output", "final.glb"}, r)) rc |= fail("flag-after-positional-output", "parse failed");
        else if (r.output != "final.glb") rc |= fail("flag-after-positional-output", "flag did not win");
        else std::printf("PASS flag-after-positional-output\n");

        trellis::TrellisParams s;
        if (!run_args({"trellis-cli", "--output", "final.glb", "front.png", "early.glb"}, s)) rc |= fail("positional-after-flag-output", "parse failed");
        else if (s.output != "early.glb") rc |= fail("positional-after-flag-output", "positional did not win");
        else std::printf("PASS positional-after-flag-output\n");

        trellis::TrellisParams t;
        if (!run_args({"trellis-cli", "--image", "flag.png", "front.png", "model.glb"}, t)) rc |= fail("positional-after-flag-image", "parse failed");
        else if (t.image != "front.png") rc |= fail("positional-after-flag-image", "positional did not win");
        else std::printf("PASS positional-after-flag-image\n");

        trellis::TrellisParams u;
        if (!run_args({"trellis-cli", "front.png", "model.glb", "--image", "flag.png"}, u)) rc |= fail("flag-after-positional-image", "parse failed");
        else if (u.image != "flag.png") rc |= fail("flag-after-positional-image", "flag did not win");
        else std::printf("PASS flag-after-positional-image\n");
    }

    if (which == "all" || which == "canonical") {
        trellis::TrellisParams p;
        const std::vector<std::string> words = {
            "trellis-cli", "--view", "right=right.png", "--view", "left=left.png",
            "--view", "back=back.png", "--view", "front=front.png",
            "--mv-mode", "shape-directional", "model.glb"
        };
        if (!run_args(words, p)) rc |= fail("canonical-order", "parse failed");
        else if (p.multiview.views.size() != 4) rc |= fail("canonical-order", "wrong view count");
        else if (p.multiview.views[0].slot != trellis::ViewSlot::Front || p.multiview.views[0].path != "front.png" ||
                 p.multiview.views[1].slot != trellis::ViewSlot::Back || p.multiview.views[1].path != "back.png" ||
                 p.multiview.views[2].slot != trellis::ViewSlot::Left || p.multiview.views[2].path != "left.png" ||
                 p.multiview.views[3].slot != trellis::ViewSlot::Right || p.multiview.views[3].path != "right.png")
            rc |= fail("canonical-order", "slots not canonical");
        else if (p.output != "model.glb" || !p.image.empty() ||
                 p.multiview.mode != trellis::MultiviewMode::ShapeDirectional)
            rc |= fail("canonical-order", "wrong output or mode");
        else std::printf("PASS canonical-order\n");
    }

    if (which == "all" || which == "single") {
        trellis::TrellisParams p;
        if (!run_args({"trellis-cli", "--view", "front=front.png", "--mv-mode", "uniform", "model.glb"}, p))
            rc |= fail("named-front-bypass", "parse failed");
        else if (p.image != "front.png" || p.multiview.active() || p.output != "model.glb")
            rc |= fail("named-front-bypass", "did not route to legacy input");
        else std::printf("PASS named-front-bypass\n");
    }

    if (which == "all" || which == "invalid") {
        rc |= expect_reject("duplicate-view", {"trellis-cli", "--view", "front=a.png", "--view", "front=b.png", "--view", "back=c.png"});
        rc |= expect_reject("unknown-slot", {"trellis-cli", "--view", "top=a.png", "--view", "front=b.png"});
        rc |= expect_reject("missing-equals", {"trellis-cli", "--view", "front"});
        rc |= expect_reject("empty-view-path", {"trellis-cli", "--view", "front=", "--view", "back=b.png"});
        rc |= expect_reject("mixed-image-flag", {"trellis-cli", "--image", "a.png", "--view", "front=f.png", "--view", "back=b.png"});
        rc |= expect_reject("mixed-positional-image", {"trellis-cli", "--view", "front=f.png", "--view", "back=b.png", "in.png", "out.glb"});
        rc |= expect_reject("pixal3d-multiview", {"trellis-cli", "--model", "pixal3d", "--view", "front=f.png", "--view", "back=b.png"});
        rc |= expect_reject("mode-without-views", {"trellis-cli", "--mv-mode", "uniform"});
        rc |= expect_reject("lone-non-front", {"trellis-cli", "--view", "back=b.png"});
        rc |= expect_reject("unknown-mode", {"trellis-cli", "--view", "front=f.png", "--view", "back=b.png", "--mv-mode", "magic"});
    }

    if (which != "all" && which != "legacy" && which != "canonical" && which != "single" && which != "invalid")
        rc |= fail("case", "unknown case " + which);
    if (rc) return 1;
    std::printf("PASS multiview argument contracts\n");
    return 0;
}
