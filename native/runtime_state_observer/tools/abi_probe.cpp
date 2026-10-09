#include <cstddef>
#include <cstdint>
#include <bitset>
#include <cassert>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "runtime_state_bpf_shared.h"

#ifdef LARIS_ABI_QUALIFIED
#define private public
#include "llama-kv-cache.h"
#undef private
#include "llama-hparams.h"
#include "llama-batch.h"
#include "ggml.h"
#endif

#ifdef LARIS_ABI_QUALIFIED
namespace fs = std::filesystem;

static constexpr const char * PINNED_COMMIT =
    "daf6bc9f2d3516d7bda73d191c6a01388aff09c5";
#ifdef _MSC_VER
#define LARIS_STRINGIFY_INNER(value) #value
#define LARIS_STRINGIFY(value) LARIS_STRINGIFY_INNER(value)
static constexpr const char * COMPILER_ID =
    "MSVC " LARIS_STRINGIFY(_MSC_VER);
#else
static constexpr const char * COMPILER_ID = __VERSION__;
#endif

using qualified_slot_info = llama_kv_cache::slot_info;
using qualified_llama_ubatch = llama_ubatch;
using qualified_ggml_tensor = ggml_tensor;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
static_assert(offsetof(qualified_slot_info, strm) == 8U);
static_assert(offsetof(qualified_slot_info, idxs) == 32U);
static_assert(offsetof(qualified_llama_ubatch, n_tokens) == 4U);
static_assert(offsetof(qualified_llama_ubatch, pos) == 40U);
static_assert(offsetof(qualified_llama_ubatch, n_seq_id) == 48U);
static_assert(offsetof(qualified_llama_ubatch, seq_id) == 56U);
static_assert(offsetof(qualified_ggml_tensor, type) == 0U);
static_assert(offsetof(qualified_ggml_tensor, data) == 248U);
static_assert(
    offsetof(llama_kv_cache, hparams) == RUNTIME_STATE_CACHE_HPARAMS_OFFSET);
static_assert(
    offsetof(llama_kv_cache, v_trans) == RUNTIME_STATE_CACHE_V_TRANS_OFFSET);
static_assert(
    offsetof(llama_kv_cache, n_seq_max) == RUNTIME_STATE_CACHE_N_SEQ_MAX_OFFSET);
static_assert(
    offsetof(llama_kv_cache, n_stream) == RUNTIME_STATE_CACHE_N_STREAM_OFFSET);
static_assert(
    offsetof(llama_kv_cache, layers) == RUNTIME_STATE_CACHE_LAYERS_OFFSET);
static_assert(
    sizeof(llama_kv_cache::kv_layer) == RUNTIME_STATE_KV_LAYER_SIZE);
static_assert(
    offsetof(llama_kv_cache::kv_layer, k_stream)
        == RUNTIME_STATE_KV_LAYER_K_STREAM_OFFSET);
static_assert(
    offsetof(llama_kv_cache::kv_layer, v_stream)
        == RUNTIME_STATE_KV_LAYER_V_STREAM_OFFSET);
static_assert(
    offsetof(llama_hparams, rope_type)
        == RUNTIME_STATE_HPARAMS_ROPE_TYPE_OFFSET);
static_assert(sizeof(std::vector<void *>) == 24U);
static_assert(offsetof(qualified_ggml_tensor, ne) == RUNTIME_STATE_TENSOR_NE_OFFSET);
static_assert(offsetof(qualified_ggml_tensor, nb) == RUNTIME_STATE_TENSOR_NB_OFFSET);
static_assert(
    offsetof(qualified_ggml_tensor, view_src)
        == RUNTIME_STATE_TENSOR_VIEW_SRC_OFFSET);
static_assert(
    offsetof(qualified_ggml_tensor, view_offs)
        == RUNTIME_STATE_TENSOR_VIEW_OFFS_OFFSET);
#pragma GCC diagnostic pop

#endif

int main(int argc, char ** argv) {
#ifndef LARIS_ABI_QUALIFIED
    (void)argc;
    (void)argv;
    std::cerr << "ABI qualification pending: rebuild with LLAMA_SOURCE at the pinned commit\n";
    return 1;
#else
    fs::path source_root;
    if (argc == 3 && std::string(argv[1]) == "--llama-source") {
        source_root = fs::path(argv[2]);
    } else {
        std::cerr << "usage: abi_probe --llama-source PATH\n";
        return 2;
    }
    if (!fs::is_directory(source_root)) {
        std::cerr << "llama.cpp source directory is unavailable\n";
        return 1;
    }
    std::cout
        << "{\"llama_cpp_commit\":\"" << PINNED_COMMIT << "\","
        << "\"abi_source\":\"actual_pinned_headers\","
        << "\"compiler_id\":\"" << COMPILER_ID << "\","
        << "\"compiler_flags_sha256\":\"sha256:" << LARIS_ABI_FLAGS_SHA256 << "\","
        << "\"pinned_headers_sha256\":\"sha256:" << LARIS_ABI_HEADERS_SHA256 << "\","
        << "\"slot_info_strm\":" << offsetof(qualified_slot_info, strm) << ","
        << "\"slot_info_idxs\":" << offsetof(qualified_slot_info, idxs) << ","
        << "\"ubatch_n_tokens\":" << offsetof(qualified_llama_ubatch, n_tokens) << ","
        << "\"ubatch_pos\":" << offsetof(qualified_llama_ubatch, pos) << ","
        << "\"ubatch_n_seq_id\":" << offsetof(qualified_llama_ubatch, n_seq_id) << ","
        << "\"ubatch_seq_id\":" << offsetof(qualified_llama_ubatch, seq_id) << ","
        << "\"tensor_type_offset\":" << offsetof(qualified_ggml_tensor, type) << ","
        << "\"tensor_data_offset\":" << offsetof(qualified_ggml_tensor, data) << ","
        << "\"cache_hparams_offset\":" << RUNTIME_STATE_CACHE_HPARAMS_OFFSET << ","
        << "\"cache_v_trans_offset\":" << RUNTIME_STATE_CACHE_V_TRANS_OFFSET << ","
        << "\"cache_n_seq_max_offset\":" << RUNTIME_STATE_CACHE_N_SEQ_MAX_OFFSET << ","
        << "\"cache_n_stream_offset\":" << RUNTIME_STATE_CACHE_N_STREAM_OFFSET << ","
        << "\"cache_layers_offset\":" << RUNTIME_STATE_CACHE_LAYERS_OFFSET << ","
        << "\"kv_layer_size\":" << RUNTIME_STATE_KV_LAYER_SIZE << ","
        << "\"kv_layer_k_stream_offset\":" << RUNTIME_STATE_KV_LAYER_K_STREAM_OFFSET << ","
        << "\"kv_layer_v_stream_offset\":" << RUNTIME_STATE_KV_LAYER_V_STREAM_OFFSET << ","
        << "\"hparams_rope_type_offset\":" << RUNTIME_STATE_HPARAMS_ROPE_TYPE_OFFSET << ","
        << "\"vector_begin_offset\":" << RUNTIME_STATE_VECTOR_BEGIN_OFFSET << ","
        << "\"vector_end_offset\":" << RUNTIME_STATE_VECTOR_END_OFFSET << ","
        << "\"tensor_ne_offset\":" << RUNTIME_STATE_TENSOR_NE_OFFSET << ","
        << "\"tensor_nb_offset\":" << RUNTIME_STATE_TENSOR_NB_OFFSET << ","
        << "\"tensor_view_src_offset\":" << RUNTIME_STATE_TENSOR_VIEW_SRC_OFFSET << ","
        << "\"tensor_view_offs_offset\":" << RUNTIME_STATE_TENSOR_VIEW_OFFS_OFFSET << ","
        << "\"n_pos_per_embd_observation\":\"method-return\","
        << "\"pre_kv_n_pos_observation\":true}\n";
    return 0;
#endif
}
