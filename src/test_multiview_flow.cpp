#include "multiview.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace {

int fail(const char* test, const std::string& detail) {
    std::fprintf(stderr, "FAIL %s: %s\n", test, detail.c_str());
    return 1;
}

bool close_enough(const std::vector<float>& a, const std::vector<double>& b, double eps = 1e-6) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (!std::isfinite(a[i]) || std::fabs((double)a[i] - b[i]) > eps) return false;
    return true;
}

std::vector<float> linear_forward(const std::vector<float>& x, float, const trellis::FlowCond& c) {
    const float cval = c.cond ? c.cond[0] : 0.0f;
    std::vector<float> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = 0.25f * x[i] + cval;
    return out;
}

void double_rescale(std::vector<double>& pred, const std::vector<double>& pos,
                    const std::vector<double>& sample, float t, float gr, float sm) {
    const double a = 1.0 - (double)sm;
    const double b = (double)sm + a * (double)t;
    const std::size_t n = pred.size();
    double mp = 0.0, mc = 0.0;
    std::vector<double> xp(n), xc(n);
    for (std::size_t k = 0; k < n; ++k) {
        xp[k] = a * sample[k] - b * pos[k];
        xc[k] = a * sample[k] - b * pred[k];
        mp += xp[k]; mc += xc[k];
    }
    mp /= (double)n; mc /= (double)n;
    double vp = 0.0, vc = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        vp += (xp[k] - mp) * (xp[k] - mp);
        vc += (xc[k] - mc) * (xc[k] - mc);
    }
    double ratio = vc > 0 ? std::sqrt(vp / (double)(n - 1)) / std::sqrt(vc / (double)(n - 1)) : 1.0;
    if (!std::isfinite(ratio)) ratio = 1.0;
    if (ratio < 0.2) ratio = 0.2;
    if (ratio > 5.0) ratio = 5.0;
    for (std::size_t k = 0; k < n; ++k) {
        const double x0r = xc[k] * ratio;
        const double x0 = gr * x0r + (1.0 - gr) * xc[k];
        pred[k] = (a * sample[k] - x0) / b;
    }
}

std::vector<double> oracle_sample(std::vector<double> sample, const std::vector<double>& cvals,
                                  const std::vector<double>& w, std::size_t channels,
                                  const trellis::SamplerParams& sp, float gr,
                                  bool per_view_rescale) {
    const std::size_t n = sample.size();
    const std::size_t tokens = n / channels;
    const std::size_t views = cvals.size();
    auto scaled = [&](float t) { return sp.rescale_t * t / (1.0f + (sp.rescale_t - 1.0f) * t); };
    for (int step = 0; step < sp.steps; ++step) {
        const float t = scaled(1.0f - (float)step / sp.steps);
        const float tprev = scaled(1.0f - (float)(step + 1) / sp.steps);
        const float gs = (sp.gi0 <= t && t <= sp.gi1) ? sp.guidance_strength : 1.0f;
        std::vector<double> pos(n, 0.0), neg(n, 0.0), pred(n, 0.0);
        for (std::size_t k = 0; k < n; ++k) neg[k] = 0.25 * sample[k];
        if (gs == 1.0f) {
            for (std::size_t k = 0; k < n; ++k) {
                const std::size_t token = k / channels;
                for (std::size_t v = 0; v < views; ++v)
                    pos[k] += w[token * views + v] * (0.25 * sample[k] + cvals[v]);
            }
            pred = pos;
        } else if (gs == 0.0f) {
            pred = neg;
        } else if (per_view_rescale) {
            for (std::size_t v = 0; v < views; ++v) {
                std::vector<double> one(n), guided(n);
                for (std::size_t k = 0; k < n; ++k) {
                    one[k] = 0.25 * sample[k] + cvals[v];
                    guided[k] = gs * one[k] + (1.0 - (double)gs) * neg[k];
                }
                if (gr > 0) double_rescale(guided, one, sample, t, gr, sp.sigma_min);
                for (std::size_t k = 0; k < n; ++k) pred[k] += w[(k / channels) * views + v] * guided[k];
            }
        } else {
            for (std::size_t k = 0; k < n; ++k) {
                const std::size_t token = k / channels;
                for (std::size_t v = 0; v < views; ++v)
                    pos[k] += w[token * views + v] * (0.25 * sample[k] + cvals[v]);
                pred[k] = gs * pos[k] + (1.0 - (double)gs) * neg[k];
            }
            if (gr > 0) double_rescale(pred, pos, sample, t, gr, sp.sigma_min);
        }
        if (tokens == 0) return {};
        for (std::size_t k = 0; k < n; ++k) sample[k] -= (double)(t - tprev) * pred[k];
    }
    return sample;
}

int expected_calls(const trellis::SamplerParams& sp, std::size_t views) {
    int calls = 0;
    auto scaled = [&](float t) { return sp.rescale_t * t / (1.0f + (sp.rescale_t - 1.0f) * t); };
    for (int step = 0; step < sp.steps; ++step) {
        const float t = scaled(1.0f - (float)step / sp.steps);
        const float gs = (sp.gi0 <= t && t <= sp.gi1) ? sp.guidance_strength : 1.0f;
        if (gs == 1.0f) calls += (int)views;
        else if (gs == 0.0f) calls += 1;
        else calls += (int)views + 1;
    }
    return calls;
}

trellis::MultiviewSpec two_views(trellis::MultiviewMode mode) {
    trellis::MultiviewSpec spec;
    spec.views.push_back({trellis::ViewSlot::Front, "front.png"});
    spec.views.push_back({trellis::ViewSlot::Back, "back.png"});
    spec.mode = mode;
    return spec;
}

int test_single() {
    trellis::SamplerParams sp; sp.steps = 3;
    std::vector<float> legacy = {0.2f, -0.4f, 0.7f};
    std::vector<float> candidate = legacy;
    const float c[3] = {0.4f, -0.2f, 0.9f};
    const float zero[3] = {0, 0, 0};
    std::vector<std::vector<float>> t1, t2;
    auto a = trellis::sample_flow(linear_forward, legacy, trellis::FlowCond(c), trellis::FlowCond(zero), sp, &t1);
    std::string error;
    if (!trellis::sample_flow_multiview(linear_forward, candidate, {trellis::FlowCond(c)},
                                        trellis::FlowCond(zero), 3, 1, {}, sp, &t2, &error))
        return fail("single", error);
    if (a != candidate || t1 != t2) return fail("single", "trace or sample differs");
    std::printf("PASS single legacy sampler is bitwise identical\n");
    return 0;
}

int test_weights() {
    const auto dense = two_views(trellis::MultiviewMode::SsShapeDirectional);
    std::vector<float> w;
    std::string error;
    if (!trellis::multiview_weights(dense, trellis::FlowStage::Ss, nullptr, 16, w, &error))
        return fail("weights", error);
    if (w.size() != 2u * 4096u) return fail("weights", "wrong dense size");
    for (int token = 0; token < 4096; ++token) {
        const int x = token / 256, y = (token / 16) % 16, z = token % 16;
        const double qx = (x + 0.5) / 16.0 - 0.5;
        const double qy = (y + 0.5) / 16.0 - 0.5;
        const double radius = std::sqrt(qx * qx + qy * qy);
        const double ux = radius > 1e-6 ? qx / radius : 0.0;
        const double uy = radius > 1e-6 ? qy / radius : 0.0;
        const double front = 0.25 + std::max(0.0, ux * 0.0 + uy * -1.0);
        const double back = 0.25 + std::max(0.0, ux * 0.0 + uy * 1.0);
        const double expected_front = front / (front + back);
        const double expected_back = back / (front + back);
        if (std::fabs((double)w[token * 2] - expected_front) > 1e-6 ||
            std::fabs((double)w[token * 2 + 1] - expected_back) > 1e-6)
            return fail("weights", "dense coordinate inversion mismatch");
    }
    std::vector<std::array<int,3>> center = {{{7,7,7}}};
    if (!trellis::multiview_weights(dense, trellis::FlowStage::Shape, &center, 15, w, &error))
        return fail("weights", error);
    if (std::fabs(w[0] - 0.5f) > 1e-6f || std::fabs(w[1] - 0.5f) > 1e-6f)
        return fail("weights", "zero-radius weights not uniform");
    const auto uniform = two_views(trellis::MultiviewMode::Uniform);
    if (!trellis::multiview_weights(uniform, trellis::FlowStage::Ss, nullptr, 16, w, &error))
        return fail("weights", error);
    if (std::fabs(w[0] - 0.5f) > 1e-6f || std::fabs(w[8191] - 0.5f) > 1e-6f)
        return fail("weights", "uniform weights wrong");
    std::printf("PASS all dense coordinates, center and uniform weights\n");
    return 0;
}

int test_sparse() {
    trellis::MultiviewSpec spec;
    spec.views.push_back({trellis::ViewSlot::Front, "f"});
    spec.views.push_back({trellis::ViewSlot::Left, "l"});
    spec.views.push_back({trellis::ViewSlot::Right, "r"});
    spec.mode = trellis::MultiviewMode::ShapeDirectional;
    for (int grid : {32, 64, 96}) {
        const std::vector<std::array<int,3>> coords = {
            {{0,0,0}}, {{grid-1,grid-1,grid-1}}, {{grid/2,grid/2,grid/2}}
        };
        const std::vector<std::array<int,3>> permuted = {
            {{grid/2,grid/2,grid/2}}, {{0,0,0}}, {{grid-1,grid-1,grid-1}}
        };
        std::vector<float> w, wp;
        std::string error;
        if (!trellis::multiview_weights(spec, trellis::FlowStage::Shape, &coords, grid, w, &error)) return fail("sparse", error);
        if (!trellis::multiview_weights(spec, trellis::FlowStage::Shape, &permuted, grid, wp, &error)) return fail("sparse", error);
        for (int p = 0; p < 3; ++p) {
            double sum = 0.0;
            for (int v = 0; v < 3; ++v) {
                sum += w[p*3+v];
                if (w[p*3+v] != wp[((p+1)%3)*3+v]) return fail("sparse", "permutation changed weights");
            }
            if (std::fabs(sum - 1.0) > 1e-6) return fail("sparse", "weights do not sum to one");
        }
        if (!(w[0*3+1] > w[0*3+2])) return fail("sparse", "left bias missing at boundary");
        if (!(w[1*3+2] > w[1*3+1])) return fail("sparse", "right bias missing at boundary");
    }
    std::printf("PASS sparse grids 32/64/96, boundaries and permutation\n");
    return 0;
}

int test_channels() {
    const std::size_t channels = 8, tokens = 4, views = 2;
    const float c0[1] = {1.0f}, c1[1] = {-1.0f}, zero[1] = {0.0f};
    const std::vector<float> weights = {0.25f,0.75f, 0.5f,0.5f, 0.75f,0.25f, 1.0f,0.0f};
    std::vector<float> sample(channels * tokens, 0.0f), actual = sample;
    auto channel_forward = [&](const std::vector<float>&, float, const trellis::FlowCond& c) {
        std::vector<float> out(channels * tokens);
        for (std::size_t k = 0; k < out.size(); ++k) out[k] = (float)(k % channels) + 0.01f * (float)(k / channels) + c.cond[0];
        return out;
    };
    trellis::SamplerParams sp; sp.steps = 1; sp.guidance_strength = 1.0f;
    std::string error;
    if (!trellis::sample_flow_multiview(channel_forward, actual,
            {trellis::FlowCond(c0), trellis::FlowCond(c1)}, trellis::FlowCond(zero),
            1, channels, weights, sp, nullptr, &error)) return fail("channels", error);
    std::vector<double> expected(channels * tokens);
    for (std::size_t j = 0; j < tokens; ++j) for (std::size_t c = 0; c < channels; ++c)
        expected[j * channels + c] = -1.0 * (weights[j*2] * ((double)c + 0.01 * j + 1.0) + weights[j*2+1] * ((double)c + 0.01 * j - 1.0));
    if (!close_enough(actual, expected)) return fail("channels", "token-major scalar indexing mismatch");
    std::printf("PASS 8-channel SS and token-major weight addressing\n");
    return 0;
}

int test_sparse_sampling() {
    const std::size_t channels = 32;
    const float cvals[4] = {0.003f, -0.002f, 0.007f, -0.005f};
    const float zero[1] = {0.0f};
    trellis::MultiviewSpec spec;
    spec.views.push_back({trellis::ViewSlot::Front, "f"});
    spec.views.push_back({trellis::ViewSlot::Back, "b"});
    spec.views.push_back({trellis::ViewSlot::Left, "l"});
    spec.views.push_back({trellis::ViewSlot::Right, "r"});
    spec.mode = trellis::MultiviewMode::ShapeDirectional;
    const std::array<std::size_t,4> perm = {2, 0, 3, 1};
    for (int grid : {32, 64, 96}) {
        const std::vector<std::array<int,3>> coords = {
            {{0,0,0}}, {{grid-1,grid/2,grid/2}}, {{grid/2,grid-1,grid/2}}, {{grid/2,grid/2,grid-1}}
        };
        std::vector<std::array<int,3>> permuted;
        for (std::size_t j : perm) permuted.push_back(coords[j]);
        std::vector<float> w, wp;
        std::string error;
        if (!trellis::multiview_weights(spec, trellis::FlowStage::Shape, &coords, grid, w, &error)) return fail("sparse-sampling", error);
        if (!trellis::multiview_weights(spec, trellis::FlowStage::Shape, &permuted, grid, wp, &error)) return fail("sparse-sampling", error);
        const std::size_t tokens = coords.size();
        for (std::size_t j = 0; j < tokens; ++j) for (std::size_t v = 0; v < 4; ++v)
            if (std::fabs(wp[j*4+v] - w[perm[j]*4+v]) > 1e-6f) return fail("sparse-sampling", "weights did not follow coordinate permutation");
        std::vector<float> sample(channels * tokens), sample_p(channels * tokens);
        for (std::size_t j = 0; j < tokens; ++j) for (std::size_t c = 0; c < channels; ++c)
            sample[j*channels+c] = 0.001f * (float)(c + 1) + 0.01f * (float)(j + 1);
        for (std::size_t j = 0; j < tokens; ++j) for (std::size_t c = 0; c < channels; ++c)
            sample_p[j*channels+c] = sample[perm[j]*channels+c];
        auto forward = [&](const std::vector<float>& x, float, const trellis::FlowCond& cond) {
            std::vector<float> out(channels * tokens);
            for (std::size_t k = 0; k < out.size(); ++k)
                out[k] = 0.25f * x[k] + cond.cond[0];
            return out;
        };
        std::vector<trellis::FlowCond> conds;
        for (std::size_t i = 0; i < 4; ++i) conds.push_back(trellis::FlowCond(cvals + i));
        trellis::SamplerParams sp; sp.steps = 1; sp.guidance_strength = 0.5f;
        sp.gi0 = 0.0f; sp.gi1 = 2.0f; sp.guidance_rescale = 0.0f;
        const std::vector<float> initial = sample, initial_p = sample_p;
        std::vector<float> actual = sample, actual_p = sample_p;
        if (!trellis::sample_flow_multiview(forward, actual, conds, trellis::FlowCond(zero), 1, channels, w, sp, nullptr, &error)) return fail("sparse-sampling", error);
        if (!trellis::sample_flow_multiview(forward, actual_p, conds, trellis::FlowCond(zero), 1, channels, wp, sp, nullptr, &error)) return fail("sparse-sampling", error);
        std::vector<double> expected(channels * tokens), expected_p(channels * tokens), permuted_actual(channels * tokens);
        for (std::size_t j = 0; j < tokens; ++j) for (std::size_t c = 0; c < channels; ++c) {
            const std::size_t k = j * channels + c;
            double cond_blend = 0.0, cond_blend_p = 0.0;
            for (std::size_t v = 0; v < 4; ++v) {
                cond_blend += w[j*4+v] * cvals[v];
                cond_blend_p += wp[j*4+v] * cvals[v];
            }
            const double pos = 0.25 * initial[k] + cond_blend;
            const double neg = 0.25 * initial[k];
            const double pos_p = 0.25 * initial_p[k] + cond_blend_p;
            const double neg_p = 0.25 * initial_p[k];
            expected[k] = (double)initial[k] - (0.5 * pos + 0.5 * neg);
            expected_p[k] = (double)initial_p[k] - (0.5 * pos_p + 0.5 * neg_p);
            permuted_actual[j*channels+c] = (double)actual[perm[j]*channels+c];
        }
        if (!close_enough(actual, expected)) {
            double max_diff = 0.0;
            for (std::size_t k = 0; k < actual.size(); ++k) max_diff = std::max(max_diff, std::fabs((double)actual[k] - expected[k]));
            return fail("sparse-sampling", "32-channel oracle mismatch max=" + std::to_string(max_diff));
        }
        if (!close_enough(actual_p, expected_p)) {
            double max_diff = 0.0;
            for (std::size_t k = 0; k < actual_p.size(); ++k) max_diff = std::max(max_diff, std::fabs((double)actual_p[k] - expected_p[k]));
            return fail("sparse-sampling", "token permutation mismatch max=" + std::to_string(max_diff));
        }
        if (!close_enough(actual_p, permuted_actual)) return fail("sparse-sampling", "sampled tokens did not follow permutation");
        const std::size_t probe_tokens[2] = {0, 2};
        const std::size_t probe_channels[2] = {0, 17};
        for (int probe = 0; probe < 2; ++probe) {
            const std::size_t k = probe_tokens[probe] * channels + probe_channels[probe];
            std::vector<float> perturbed = initial;
            perturbed[k] += 1.0f;
            std::vector<float> perturbed_out = perturbed;
            error.clear();
            if (!trellis::sample_flow_multiview(forward, perturbed_out, conds, trellis::FlowCond(zero), 1, channels, w, sp, nullptr, &error))
                return fail("sparse-sampling", error);
            if (std::fabs((double)perturbed_out[k] - actual[k]) < 0.5)
                return fail("sparse-sampling", "token/channel sample sentinel did not affect output");
        }
    }
    std::printf("PASS 32-channel sparse sampling at grids 32/64/96 with token permutation\n");
    return 0;
}

int test_partial_views() {
    const std::array<trellis::ViewSlot,4> slots = {
        trellis::ViewSlot::Front, trellis::ViewSlot::Back,
        trellis::ViewSlot::Left, trellis::ViewSlot::Right
    };
    const std::vector<std::array<int,3>> coords = {{{4,4,4}}, {{8,8,8}}, {{12,12,12}}};
    for (int mask = 1; mask < 16; ++mask) {
        if (!(mask & 1)) continue;
        trellis::MultiviewSpec spec;
        for (int bit = 0; bit < 4; ++bit) if (mask & (1 << bit))
            spec.views.push_back({slots[bit], "view" + std::to_string(bit)});
        spec.mode = trellis::MultiviewMode::SsShapeDirectional;
        std::string error;
        std::vector<float> weights;
        if (!trellis::multiview_weights(spec, trellis::FlowStage::Ss, nullptr, 16, weights, &error)) return fail("partial-views", error);
        if (weights.size() != spec.views.size() * 4096) return fail("partial-views", "dense size mismatch");
        for (float value : weights) if (!std::isfinite(value) || value < 0.0f) return fail("partial-views", "invalid dense weight");
        for (std::size_t token = 0; token < 4096; ++token) {
            double sum = 0.0;
            for (std::size_t v = 0; v < spec.views.size(); ++v) sum += weights[token*spec.views.size()+v];
            if (std::fabs(sum - 1.0) > 1e-6) return fail("partial-views", "dense weights not normalized");
        }
        if (!trellis::multiview_weights(spec, trellis::FlowStage::Shape, &coords, 16, weights, &error)) return fail("partial-views", error);
        for (std::size_t token = 0; token < coords.size(); ++token) {
            double sum = 0.0;
            for (std::size_t v = 0; v < spec.views.size(); ++v) {
                const float value = weights[token*spec.views.size()+v];
                if (!std::isfinite(value) || value < 0.0f) return fail("partial-views", "invalid sparse weight");
                sum += value;
            }
            if (std::fabs(sum - 1.0) > 1e-6) return fail("partial-views", "sparse weights not normalized");
        }
    }
    std::printf("PASS all front-containing partial view sets from 1 to 4 views\n");
    return 0;
}

int test_guidance() {
    const std::size_t n = 3, channels = 1;
    const float c0[n] = {0.4f, -0.2f, 0.9f};
    const float c1[n] = {-0.7f, 0.3f, 0.1f};
    const float zero[n] = {0, 0, 0};
    const std::vector<float> weights = {0.5f,0.5f, 0.25f,0.75f, 0.75f,0.25f};
    const std::vector<double> oracle_w(weights.begin(), weights.end());
    const struct Schedule { float g, gi0, gi1, gr, rt; int steps; } schedules[] = {
        {0.0f, 0.0f, 2.0f, 0.5f, 1.0f, 4},
        {1.0f, 0.0f, 2.0f, 0.5f, 1.0f, 4},
        {7.5f, 0.6f, 1.0f, 0.7f, 5.0f, 4},
        {7.5f, 0.6f, 1.0f, 0.5f, 3.0f, 4},
        {7.5f, 0.6f, 0.6f, 0.7f, 1.0f, 5},
    };
    for (const auto& s : schedules) {
        trellis::SamplerParams sp; sp.steps = s.steps; sp.guidance_strength = s.g;
        sp.gi0 = s.gi0; sp.gi1 = s.gi1; sp.guidance_rescale = s.gr; sp.rescale_t = s.rt;
        std::vector<float> sample = {0.2f, -0.4f, 0.7f}, actual = sample;
        const std::vector<trellis::FlowCond> conds = {trellis::FlowCond(c0), trellis::FlowCond(c1)};
        std::string error;
        if (!trellis::sample_flow_multiview(linear_forward, actual, conds,
                trellis::FlowCond(zero), n, channels, weights, sp, nullptr, &error))
            return fail("guidance", error);
        const auto expected = oracle_sample({0.2, -0.4, 0.7}, {0.4, -0.7}, oracle_w, channels, sp, s.gr, false);
        if (!close_enough(actual, expected, 1e-6)) return fail("guidance", "CFG/rescale oracle mismatch");
        int calls = 0;
        auto counting = [&](const std::vector<float>& x, float t, const trellis::FlowCond& c) {
            ++calls; return linear_forward(x, t, c);
        };
        std::vector<float> counted = sample;
        if (!trellis::sample_flow_multiview(counting, counted, conds,
                trellis::FlowCond(zero), n, channels, weights, sp, nullptr, &error))
            return fail("guidance", error);
        if (calls != expected_calls(sp, 2)) return fail("guidance", "unexpected forward count");
    }
    trellis::SamplerParams sp; sp.steps = 2; sp.guidance_strength = 7.5f;
    sp.gi0 = 0.0f; sp.gi1 = 2.0f; sp.guidance_rescale = 0.5f; sp.rescale_t = 1.0f;
    std::vector<float> sample = {0.2f, -0.4f, 0.7f}, actual = sample;
    const std::vector<trellis::FlowCond> conds = {trellis::FlowCond(c0), trellis::FlowCond(c1)};
    std::string error;
    if (!trellis::sample_flow_multiview(linear_forward, actual, conds,
            trellis::FlowCond(zero), n, channels, weights, sp, nullptr, &error)) return fail("guidance", error);
    const auto once = oracle_sample({0.2, -0.4, 0.7}, {0.4, -0.7}, oracle_w, channels, sp, 0.5f, false);
    const auto wrong = oracle_sample({0.2, -0.4, 0.7}, {0.4, -0.7}, oracle_w, channels, sp, 0.5f, true);
    if (!close_enough(actual, once, 1e-6)) return fail("guidance", "rescale-once mismatch");
    if (close_enough(actual, wrong, 1e-4)) return fail("guidance", "wrong per-view rescale matched");
    std::printf("PASS guidance 0/1/7.5, SS/shape schedules, boundaries and call counts\n");
    return 0;
}

int test_invalid() {
    const float c[3] = {1,1,1};
    const float bad_c[3] = {1, std::numeric_limits<float>::quiet_NaN(), 1};
    const float zero[3] = {0,0,0};
    const float inf_c[3] = {1.0f, std::numeric_limits<float>::infinity(), 1.0f};
    trellis::SamplerParams sp; sp.steps = 1;
    std::string error;
    std::vector<float> sample = {0.1f, 0.2f, 0.3f};
    std::vector<trellis::FlowCond> conds = {trellis::FlowCond(c), trellis::FlowCond(c)};
    std::vector<float> weights(6, 0.5f);
    std::vector<std::size_t> lengths{2, 3};
    error.clear();
    std::vector<float> empty_sample;
    if (trellis::sample_flow_multiview(linear_forward, empty_sample, conds, trellis::FlowCond(zero), 3, 1, weights, sp, nullptr, &error)) return fail("invalid", "empty sample accepted");
    if (trellis::sample_flow_multiview(linear_forward, sample, {}, trellis::FlowCond(zero), 3, 1, weights, sp, nullptr, &error)) return fail("invalid", "zero views accepted");
    error.clear();
    if (trellis::sample_flow_multiview(linear_forward, sample, conds, trellis::FlowCond(zero), 3, 1, weights, sp, nullptr, &error, &lengths)) return fail("invalid", "cond length mismatch accepted");
    error.clear();
    std::vector<std::size_t> long_lengths{4, 4};
    if (trellis::sample_flow_multiview(linear_forward, sample, conds, trellis::FlowCond(zero), 3, 1, weights, sp, nullptr, &error, &long_lengths)) return fail("invalid", "long condition accepted");
    error.clear();
    weights.resize(5);
    if (trellis::sample_flow_multiview(linear_forward, sample, conds, trellis::FlowCond(zero), 3, 1, weights, sp, nullptr, &error)) return fail("invalid", "weight size mismatch accepted");
    weights.assign(6, 0.5f);
    error.clear();
    if (trellis::sample_flow_multiview(linear_forward, sample, conds, trellis::FlowCond(zero), 3, 2, weights, sp, nullptr, &error)) return fail("invalid", "invalid channel layout accepted");
    error.clear();
    if (trellis::sample_flow_multiview(linear_forward, sample, {trellis::FlowCond(bad_c), trellis::FlowCond(c)}, trellis::FlowCond(zero), 3, 1, weights, sp, nullptr, &error)) return fail("invalid", "nonfinite condition accepted");
    sample[1] = std::numeric_limits<float>::quiet_NaN();
    error.clear();
    if (trellis::sample_flow_multiview(linear_forward, sample, conds, trellis::FlowCond(zero), 3, 1, weights, sp, nullptr, &error)) return fail("invalid", "nonfinite sample accepted");
    sample[1] = 0.2f;
    error.clear();
    auto bad_length = [](const std::vector<float>&, float, const trellis::FlowCond&) { return std::vector<float>(2, 0.0f); };
    if (trellis::sample_flow_multiview(bad_length, sample, conds, trellis::FlowCond(zero), 3, 1, weights, sp, nullptr, &error)) return fail("invalid", "malformed forward length accepted");
    error.clear();
    auto long_velocity = [](const std::vector<float>&, float, const trellis::FlowCond&) { return std::vector<float>(5, 0.0f); };
    if (trellis::sample_flow_multiview(long_velocity, sample, conds, trellis::FlowCond(zero), 3, 1, weights, sp, nullptr, &error)) return fail("invalid", "long velocity accepted");
    error.clear();
    auto bad_value = [](const std::vector<float>& x, float, const trellis::FlowCond&) {
        auto out = x; out[0] = std::numeric_limits<float>::quiet_NaN(); return out;
    };
    if (trellis::sample_flow_multiview(bad_value, sample, conds, trellis::FlowCond(zero), 3, 1, weights, sp, nullptr, &error)) return fail("invalid", "nonfinite velocity accepted");
    error.clear();
    if (trellis::sample_flow_multiview(linear_forward, sample, {trellis::FlowCond(inf_c), trellis::FlowCond(c)}, trellis::FlowCond(zero), 3, 1, weights, sp, nullptr, &error)) return fail("invalid", "infinite condition accepted");
    error.clear();
    error.clear();
    auto inf_velocity = [](const std::vector<float>& x, float, const trellis::FlowCond&) {
        auto out = x; out[2] = std::numeric_limits<float>::infinity(); return out;
    };
    if (trellis::sample_flow_multiview(inf_velocity, sample, conds, trellis::FlowCond(zero), 3, 1, weights, sp, nullptr, &error)) return fail("invalid", "infinite velocity accepted");
    error.clear();
    weights.assign(6, 0.5f);
    auto bad_negative = [&](const std::vector<float>& x, float t, const trellis::FlowCond& cond) {
        if (cond.cond == zero) {
            auto out = linear_forward(x, t, cond);
            out[0] = std::numeric_limits<float>::quiet_NaN();
            return out;
        }
        return linear_forward(x, t, cond);
    };
    trellis::SamplerParams sp0; sp0.steps = 1; sp0.guidance_strength = 7.5f;
    sp0.gi0 = 0.0f; sp0.gi1 = 2.0f; sp0.guidance_rescale = 0.5f;
    if (trellis::sample_flow_multiview(bad_negative, sample, conds, trellis::FlowCond(zero), 3, 1, weights, sp0, nullptr, &error)) return fail("invalid", "malformed negative velocity accepted");
    weights.assign(6, 0.2f);
    if (trellis::sample_flow_multiview(linear_forward, sample, conds, trellis::FlowCond(zero), 3, 1, weights, sp, nullptr, &error)) return fail("invalid", "unnormalized weights accepted");
    trellis::MultiviewSpec spec = two_views(trellis::MultiviewMode::Uniform);
    std::vector<float> out;
    error.clear();
    if (trellis::multiview_weights(spec, trellis::FlowStage::Shape, nullptr, 32, out, &error)) return fail("invalid", "missing sparse coords accepted");
    error.clear();
    std::vector<std::array<int,3>> empty_coords;
    if (trellis::multiview_weights(spec, trellis::FlowStage::Shape, &empty_coords, 32, out, &error)) return fail("invalid", "empty coordinate vector accepted");
    std::vector<std::array<int,3>> coords = {{{99,0,0}}};
    error.clear();
    if (trellis::multiview_weights(spec, trellis::FlowStage::Shape, &coords, 32, out, &error)) return fail("invalid", "out-of-grid uniform coord accepted");
    coords = {{{1,2,3}}};
    error.clear();
    if (trellis::multiview_weights(spec, trellis::FlowStage::Shape, &coords, 0, out, &error)) return fail("invalid", "zero grid accepted");
    std::printf("PASS malformed multiview inputs rejected without repair\n");
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
    if (which == "all" || which == "single") rc |= test_single();
    if (which == "all" || which == "weights" || which == "dense") rc |= test_weights();
    if (which == "all" || which == "sparse") rc |= test_sparse();
    if (which == "all" || which == "channels") rc |= test_channels();
    if (which == "all" || which == "sparse-sampling") rc |= test_sparse_sampling();
    if (which == "all" || which == "partial-views") rc |= test_partial_views();
    if (which == "all" || which == "guidance") rc |= test_guidance();
    if (which == "all" || which == "invalid") rc |= test_invalid();
    if (which != "all" && which != "single" && which != "weights" && which != "dense" &&
        which != "sparse" && which != "sparse-sampling" && which != "partial-views" &&
        which != "channels" && which != "guidance" && which != "invalid")
        rc |= fail("case", "unknown case " + which);
    if (rc) return 1;
    std::printf("PASS multiview CPU contracts\n");
    return 0;
}
