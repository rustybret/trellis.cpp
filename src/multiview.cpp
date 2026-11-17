#include "multiview.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace trellis {

namespace {

bool view_direction(ViewSlot slot, float& dx, float& dy) {
    switch (slot) {
        case ViewSlot::Front: dx = 0.0f;  dy = -1.0f; return true;
        case ViewSlot::Back:  dx = 0.0f;  dy =  1.0f; return true;
        case ViewSlot::Left:  dx = -1.0f; dy =  0.0f; return true;
        case ViewSlot::Right: dx = 1.0f;  dy =  0.0f; return true;
    }
    return false;
}

bool finite_data(const float* data, std::size_t count) {
    if (!data) return false;
    for (std::size_t i = 0; i < count; ++i) if (!std::isfinite(data[i])) return false;
    return true;
}

void set_error(std::string* error, const char* message) {
    if (error) *error = message;
}

}

const char* view_slot_name(ViewSlot slot) {
    switch (slot) {
        case ViewSlot::Front: return "front";
        case ViewSlot::Back:  return "back";
        case ViewSlot::Left:  return "left";
        case ViewSlot::Right: return "right";
    }
    return "unknown";
}

const char* multiview_mode_name(MultiviewMode mode) {
    switch (mode) {
        case MultiviewMode::Uniform:           return "uniform";
        case MultiviewMode::ShapeDirectional:  return "shape-directional";
        case MultiviewMode::SsShapeDirectional: return "ss-shape-directional";
    }
    return "unknown";
}

const std::string* MultiviewSpec::path(ViewSlot slot) const {
    for (const auto& view : views) if (view.slot == slot) return &view.path;
    return nullptr;
}

bool parse_view_arg(const std::string& value, MultiviewSpec& spec, std::string& error) {
    const std::size_t equals = value.find('=');
    if (equals == std::string::npos || equals == 0 || equals + 1 == value.size()) {
        error = "--view expects NAME=PATH";
        return false;
    }
    const std::string name = value.substr(0, equals);
    const std::string path = value.substr(equals + 1);
    ViewSlot slot;
    if (name == "front") slot = ViewSlot::Front;
    else if (name == "back") slot = ViewSlot::Back;
    else if (name == "left") slot = ViewSlot::Left;
    else if (name == "right") slot = ViewSlot::Right;
    else {
        error = "unknown view slot: " + name;
        return false;
    }
    if (spec.path(slot)) {
        error = std::string("duplicate view slot: ") + view_slot_name(slot);
        return false;
    }
    spec.views.push_back({slot, path});
    std::sort(spec.views.begin(), spec.views.end(),
              [](const ViewSpec& a, const ViewSpec& b) {
                  return static_cast<int>(a.slot) < static_cast<int>(b.slot);
              });
    return true;
}

bool parse_multiview_mode_arg(const std::string& value, MultiviewSpec& spec, std::string& error) {
    if (value == "uniform") spec.mode = MultiviewMode::Uniform;
    else if (value == "shape-directional") spec.mode = MultiviewMode::ShapeDirectional;
    else if (value == "ss-shape-directional") spec.mode = MultiviewMode::SsShapeDirectional;
    else {
        error = "unknown multiview mode: " + value;
        return false;
    }
    spec.mode_explicit = true;
    return true;
}

bool validate_multiview(const MultiviewSpec& spec, bool pixal3d, bool image_set,
                        std::string& error) {
    if (spec.views.empty()) {
        if (spec.mode_explicit) {
            error = "--mv-mode requires named --view inputs";
            return false;
        }
        return true;
    }
    if (pixal3d) {
        error = "multiview is not supported with --model pixal3d";
        return false;
    }
    if (image_set) {
        error = "--view cannot be combined with --image or an input positional";
        return false;
    }
    if (spec.views.size() > 4) {
        error = "at most four named views are accepted";
        return false;
    }
    if (!spec.path(ViewSlot::Front)) {
        error = "multiview requires a front view";
        return false;
    }
    for (const auto& view : spec.views) {
        if (view.path.empty()) {
            error = "view paths must not be empty";
            return false;
        }
    }
    return true;
}

std::vector<float> uniform_multiview_weights(std::size_t view_count, std::size_t token_count) {
    std::vector<float> weights(view_count * token_count, 0.0f);
    if (view_count == 0) return weights;
    const float value = 1.0f / static_cast<float>(view_count);
    std::fill(weights.begin(), weights.end(), value);
    return weights;
}

bool multiview_weights(const MultiviewSpec& spec, FlowStage stage,
                       const std::vector<std::array<int, 3>>* coords, int grid,
                       std::vector<float>& weights, std::string* error) {
    const std::size_t views = spec.views.size();
    if (views == 0) {
        set_error(error, "no views supplied");
        return false;
    }
    const bool sparse = stage == FlowStage::Shape;
    if (sparse && (coords == nullptr || coords->empty())) {
        set_error(error, "sparse shape stage requires current coordinates");
        return false;
    }
    if (sparse && grid <= 0) {
        set_error(error, "sparse shape stage requires a positive grid");
        return false;
    }
    const int grid_size = sparse ? grid : 16;
    const std::size_t tokens = sparse ? coords->size() : static_cast<std::size_t>(16 * 16 * 16);
    if (sparse) {
        for (const auto& c : *coords) {
            if (c[0] < 0 || c[1] < 0 || c[2] < 0 ||
                c[0] >= grid || c[1] >= grid || c[2] >= grid) {
                set_error(error, "sparse coordinate outside grid");
                return false;
            }
        }
    }
    const bool directional = spec.mode == MultiviewMode::SsShapeDirectional ||
        (spec.mode == MultiviewMode::ShapeDirectional && sparse);
    if (!directional) {
        weights = uniform_multiview_weights(views, tokens);
        return true;
    }

    std::vector<float> rx(views), ry(views);
    for (std::size_t v = 0; v < views; ++v) {
        if (!view_direction(spec.views[v].slot, rx[v], ry[v])) {
            set_error(error, "unknown view direction");
            return false;
        }
    }
    weights.assign(views * tokens, 0.0f);
    for (std::size_t token = 0; token < tokens; ++token) {
        int x, y, z;
        if (sparse) {
            x = (*coords)[token][0]; y = (*coords)[token][1]; z = (*coords)[token][2];
            if (x < 0 || y < 0 || z < 0 || x >= grid || y >= grid || z >= grid) {
                set_error(error, "sparse coordinate outside grid");
                return false;
            }
        } else {
            x = static_cast<int>(token / (16 * 16));
            y = static_cast<int>((token / 16) % 16);
            z = static_cast<int>(token % 16);
        }
        const float qx = (static_cast<float>(x) + 0.5f) / grid_size - 0.5f;
        const float qy = (static_cast<float>(y) + 0.5f) / grid_size - 0.5f;
        const float radius = std::sqrt(qx * qx + qy * qy);
        const float ux = radius > 1e-6f ? qx / radius : 0.0f;
        const float uy = radius > 1e-6f ? qy / radius : 0.0f;
        float sum = 0.0f;
        std::vector<float> activation(views);
        for (std::size_t v = 0; v < views; ++v) {
            const float dot = ux * rx[v] + uy * ry[v];
            activation[v] = 0.25f + std::max(0.0f, dot);
            if (!std::isfinite(activation[v])) {
                set_error(error, "nonfinite directional weight");
                return false;
            }
            sum += activation[v];
        }
        if (!std::isfinite(sum) || sum <= 0.0f) {
            set_error(error, "invalid directional weight sum");
            return false;
        }
        for (std::size_t v = 0; v < views; ++v) weights[token * views + v] = activation[v] / sum;
    }
    return true;
}

bool sample_flow_multiview(const FlowFwd& fwd,
                           std::vector<float>& sample,
                           const std::vector<FlowCond>& conds,
                           const FlowCond& neg_cond,
                           std::size_t cond_floats,
                           std::size_t channels,
                           const std::vector<float>& weights,
                           const SamplerParams& sp,
                           std::vector<std::vector<float>>* trace,
                           std::string* error,
                           const std::vector<std::size_t>* cond_lengths) {
    if (conds.empty()) {
        set_error(error, "no view conditions supplied");
        return false;
    }
    if (sample.empty()) {
        set_error(error, "empty sample");
        return false;
    }
    if (cond_floats == 0) {
        set_error(error, "empty conditioning vector");
        return false;
    }
    if (channels == 0 || sample.size() % channels != 0) {
        set_error(error, "invalid sample channel layout");
        return false;
    }
    const std::size_t token_count = sample.size() / channels;
    if (cond_lengths) {
        if (cond_lengths->size() != conds.size()) {
            set_error(error, "condition length table size mismatch");
            return false;
        }
        for (std::size_t length : *cond_lengths) {
            if (length != cond_floats) {
                set_error(error, "condition length mismatch");
                return false;
            }
        }
    }
    if (!finite_data(sample.data(), sample.size())) {
        set_error(error, "nonfinite input sample");
        return false;
    }
    if (!finite_data(neg_cond.cond, cond_floats) || neg_cond.proj) {
        set_error(error, "invalid negative conditioning");
        return false;
    }
    for (const auto& cond : conds) {
        if (!finite_data(cond.cond, cond_floats) || cond.proj) {
            set_error(error, "invalid view conditioning");
            return false;
        }
    }
    if (conds.size() == 1) {
        if (!weights.empty() && weights.size() != token_count) {
            set_error(error, "single-view weight size mismatch");
            return false;
        }
        sample = sample_flow(fwd, sample, conds[0], neg_cond, sp, trace);
        return true;
    }
    if (weights.size() != token_count * conds.size()) {
        set_error(error, "multiview weight size mismatch");
        return false;
    }
    for (float weight : weights) {
        if (!std::isfinite(weight) || weight < 0.0f) {
            set_error(error, "nonfinite or negative multiview weight");
            return false;
        }
    }
    for (std::size_t token = 0; token < token_count; ++token) {
        double sum = 0.0;
        for (std::size_t v = 0; v < conds.size(); ++v) sum += weights[token * conds.size() + v];
        if (!std::isfinite(sum) || std::fabs(sum - 1.0) > 1e-3) {
            set_error(error, "multiview weights do not sum to one");
            return false;
        }
    }
    const std::size_t n = sample.size();
    const std::size_t views = conds.size();
    static const bool no_fix = std::getenv("TRELLIS_NOFIX") != nullptr;
    std::vector<float> ts(sp.steps + 1);
    for (int i = 0; i <= sp.steps; ++i) {
        const float t = 1.0f - static_cast<float>(i) / sp.steps;
        ts[i] = sp.rescale_t * t / (1.0f + (sp.rescale_t - 1.0f) * t);
    }
    std::vector<float> pos(n), neg(n), pred(n), x0p(n), x0c(n);
    int forwards = 0;
    for (int step = 0; step < sp.steps; ++step) {
        const float t = ts[step], tprev = ts[step + 1];
        const float gs = (sp.gi0 <= t && t <= sp.gi1) ? sp.guidance_strength : 1.0f;
        const float tscaled = 1000.0f * t;
        if (gs == 1.0f) {
            std::fill(pos.begin(), pos.end(), 0.0f);
            for (std::size_t v = 0; v < views; ++v) {
                std::vector<float> one = fwd(sample, tscaled, conds[v]);
                ++forwards;
                if (one.size() != n) {
                    set_error(error, "forward velocity size mismatch");
                    return false;
                }
                if (!finite_data(one.data(), one.size())) {
                    set_error(error, "nonfinite view velocity");
                    return false;
                }
                for (std::size_t k = 0; k < n; ++k) pos[k] += weights[(k / channels) * views + v] * one[k];
            }
            pred = pos;
        } else {
            if (gs != 0.0f) {
                std::fill(pos.begin(), pos.end(), 0.0f);
                for (std::size_t v = 0; v < views; ++v) {
                    std::vector<float> one = fwd(sample, tscaled, conds[v]);
                    ++forwards;
                    if (one.size() != n) {
                        set_error(error, "forward velocity size mismatch");
                        return false;
                    }
                    if (!finite_data(one.data(), one.size())) {
                        set_error(error, "nonfinite view velocity");
                        return false;
                    }
                    for (std::size_t k = 0; k < n; ++k) pos[k] += weights[(k / channels) * views + v] * one[k];
                }
            }
            neg = fwd(sample, tscaled, neg_cond);
            ++forwards;
            if (neg.size() != n) {
                set_error(error, "negative velocity size mismatch");
                return false;
            }
            if (!finite_data(neg.data(), neg.size())) {
                set_error(error, "nonfinite negative velocity");
                return false;
            }
            for (std::size_t k = 0; k < n; ++k) pred[k] = gs * pos[k] + (1.0f - gs) * neg[k];
            if (sp.guidance_rescale > 0.0f && gs != 0.0f) {
                const float a = 1.0f - sp.sigma_min;
                const float b = sp.sigma_min + a * t;
                double mp = 0.0, mc = 0.0;
                for (std::size_t k = 0; k < n; ++k) {
                    x0p[k] = a * sample[k] - b * pos[k];
                    x0c[k] = a * sample[k] - b * pred[k];
                    mp += x0p[k]; mc += x0c[k];
                }
                mp /= static_cast<double>(n); mc /= static_cast<double>(n);
                double vp = 0.0, vc = 0.0;
                for (std::size_t k = 0; k < n; ++k) {
                    vp += (x0p[k] - mp) * (x0p[k] - mp);
                    vc += (x0c[k] - mc) * (x0c[k] - mc);
                }
                float ratio = 1.0f;
                if (n > 1 && vc > 0) ratio = static_cast<float>(std::sqrt(vp / (n - 1)) / std::sqrt(vc / (n - 1)));
                if (!no_fix) {
                    if (!std::isfinite(ratio)) ratio = 1.0f;
                    ratio = std::min(std::max(ratio, 0.2f), 5.0f);
                }
                const float gr = sp.guidance_rescale;
                for (std::size_t k = 0; k < n; ++k) {
                    const float x0r = x0c[k] * ratio;
                    const float x0 = gr * x0r + (1.0f - gr) * x0c[k];
                    pred[k] = (a * sample[k] - x0) / b;
                }
            }
        }
        if (!finite_data(pred.data(), pred.size())) {
            set_error(error, "nonfinite guided velocity");
            return false;
        }
        for (std::size_t k = 0; k < n; ++k) sample[k] -= (t - tprev) * pred[k];
        if (!finite_data(sample.data(), sample.size())) {
            set_error(error, "nonfinite multiview sample after Euler update");
            return false;
        }
        if (trace) trace->push_back(sample);
    }
    std::printf("      [flow] %d steps, %d forwards, %zu views\n", sp.steps, forwards, views);
    return true;
}

}
