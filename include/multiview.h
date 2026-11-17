// CLI-first TRELLIS.2 multiview contracts. This module owns named view slots,
// directional weighting and the model-free sampler used by CPU regression tests.
// It never loads a model, discovers a backend, reads a GGUF or performs inference.
#pragma once
#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "flow_runner.h"

namespace trellis {

enum class ViewSlot { Front = 0, Back = 1, Left = 2, Right = 3 };
enum class MultiviewMode { Uniform, ShapeDirectional, SsShapeDirectional };
enum class FlowStage { Ss, Shape };

const char* view_slot_name(ViewSlot slot);
const char* multiview_mode_name(MultiviewMode mode);

struct ViewSpec {
    ViewSlot slot = ViewSlot::Front;
    std::string path;
};

struct MultiviewSpec {
    // Always stored in canonical front, back, left, right order, not argv order.
    std::vector<ViewSpec> views;
    MultiviewMode mode = MultiviewMode::Uniform;
    bool mode_explicit = false;

    bool active() const { return views.size() > 1; }
    const std::string* path(ViewSlot slot) const;
};

bool parse_view_arg(const std::string& value, MultiviewSpec& spec, std::string& error);
bool parse_multiview_mode_arg(const std::string& value, MultiviewSpec& spec, std::string& error);
bool validate_multiview(const MultiviewSpec& spec, bool pixal3d, bool image_set,
                        std::string& error);

std::vector<float> uniform_multiview_weights(std::size_t view_count, std::size_t token_count);

// weights are [token * view_count + view_index]. Dense SS passes coords=nullptr and
// grid=16; sparse shape passes the current stage coordinates and its grid size.
bool multiview_weights(const MultiviewSpec& spec, FlowStage stage,
                       const std::vector<std::array<int, 3>>* coords, int grid,
                       std::vector<float>& weights, std::string* error);

// Blend one positive velocity per view on the same sample/timestep, then apply the
// shared negative, guidance rescale and one Euler update. `weights` is token-major
// [token * views + view]; `sample` is channel-major [channels * tokens]. V=1
// delegates to the legacy sampler so a named front never changes arithmetic.
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
                           const std::vector<std::size_t>* cond_lengths = nullptr);

}
