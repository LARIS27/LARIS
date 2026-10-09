#ifndef LARIS_RUNTIME_STATE_FIXTURE_H
#define LARIS_RUNTIME_STATE_FIXTURE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define RUNTIME_STATE_FIXTURE_N_STREAM 4U
#define RUNTIME_STATE_FIXTURE_N_LAYER 24U
#define RUNTIME_STATE_FIXTURE_KV_SIZE 192U
#define RUNTIME_STATE_FIXTURE_ROW_SIZE 256U
#define RUNTIME_STATE_FIXTURE_EMBEDDING_WIDTH 128U
#define RUNTIME_STATE_FIXTURE_ELEMENT_SIZE \
    (RUNTIME_STATE_FIXTURE_ROW_SIZE / RUNTIME_STATE_FIXTURE_EMBEDDING_WIDTH)
#define RUNTIME_STATE_FIXTURE_TENSOR_COUNT \
    (2U * RUNTIME_STATE_FIXTURE_N_STREAM * RUNTIME_STATE_FIXTURE_N_LAYER)

struct runtime_state_fixture_vector {
    void *begin;
    void *end;
    void *capacity_end;
};

struct runtime_state_fixture_slot_info {
    uint32_t s0;
    uint32_t s1;
    struct runtime_state_fixture_vector strm;
    struct runtime_state_fixture_vector idxs;
};

struct runtime_state_fixture_ubatch {
    uint32_t b_equal_seqs;
    uint32_t n_tokens;
    uint32_t n_seq_tokens;
    uint32_t n_seqs;
    uint32_t n_seqs_unq;
    uint32_t n_pos;
    int32_t *token;
    float *embd;
    int32_t *pos;
    int32_t *n_seq_id;
    int32_t **seq_id;
};

struct runtime_state_fixture_tensor {
    int32_t type;
    uint32_t type_padding;
    void *buffer;
    int64_t ne[4];
    size_t nb[4];
    uint8_t reserved[152];
    void *view_src;
    size_t view_offs;
    void *data;
};

struct runtime_state_fixture_cache_layer {
    uint32_t il;
    uint32_t padding;
    struct runtime_state_fixture_tensor *k;
    struct runtime_state_fixture_tensor *v;
    struct runtime_state_fixture_vector k_stream;
    struct runtime_state_fixture_vector v_stream;
};

struct runtime_state_fixture_cache {
    uint8_t base_and_model[16];
    void *hparams;
    bool v_trans;
    uint8_t scalar_padding[3];
    uint32_t n_seq_max;
    uint32_t n_stream;
    uint8_t cache_padding[236];
    struct runtime_state_fixture_vector layers;
};

struct runtime_state_fixture_buffer {
    void *base;
    size_t size;
};

struct runtime_state_fixture_context {
    void *cache_object;
    const struct runtime_state_fixture_slot_info *slot;
};

struct runtime_state_fixture_memory_context {
    int graph_status;
    uint32_t write_generation;
};

_Static_assert(
    offsetof(struct runtime_state_fixture_slot_info, strm) == 8U,
    "fixture slot stream vector offset mismatch");
_Static_assert(
    offsetof(struct runtime_state_fixture_slot_info, idxs) == 32U,
    "fixture slot index vector offset mismatch");
_Static_assert(
    offsetof(struct runtime_state_fixture_ubatch, n_tokens) == 4U,
    "fixture ubatch n_tokens offset mismatch");
_Static_assert(
    offsetof(struct runtime_state_fixture_ubatch, pos) == 40U,
    "fixture ubatch pos offset mismatch");
_Static_assert(
    offsetof(struct runtime_state_fixture_ubatch, n_seq_id) == 48U,
    "fixture ubatch n_seq_id offset mismatch");
_Static_assert(
    offsetof(struct runtime_state_fixture_ubatch, seq_id) == 56U,
    "fixture ubatch seq_id offset mismatch");
_Static_assert(
    offsetof(struct runtime_state_fixture_tensor, type) == 0U,
    "fixture tensor type offset mismatch");
_Static_assert(
    offsetof(struct runtime_state_fixture_tensor, data) == 248U,
    "fixture tensor data offset mismatch");
_Static_assert(
    offsetof(struct runtime_state_fixture_tensor, ne) == 16U,
    "fixture tensor ne offset mismatch");
_Static_assert(
    offsetof(struct runtime_state_fixture_tensor, nb) == 48U,
    "fixture tensor nb offset mismatch");
_Static_assert(
    offsetof(struct runtime_state_fixture_tensor, view_src) == 232U,
    "fixture tensor view source offset mismatch");
_Static_assert(
    offsetof(struct runtime_state_fixture_tensor, view_offs) == 240U,
    "fixture tensor view offset mismatch");
_Static_assert(
    sizeof(struct runtime_state_fixture_cache_layer) == 72U,
    "fixture cache layer size mismatch");
_Static_assert(
    offsetof(struct runtime_state_fixture_cache_layer, k_stream) == 24U,
    "fixture K stream vector offset mismatch");
_Static_assert(
    offsetof(struct runtime_state_fixture_cache_layer, v_stream) == 48U,
    "fixture V stream vector offset mismatch");
_Static_assert(
    offsetof(struct runtime_state_fixture_cache, hparams) == 16U,
    "fixture cache hparams offset mismatch");
_Static_assert(
    offsetof(struct runtime_state_fixture_cache, v_trans) == 24U,
    "fixture cache v_trans offset mismatch");
_Static_assert(
    offsetof(struct runtime_state_fixture_cache, n_seq_max) == 28U,
    "fixture cache n_seq_max offset mismatch");
_Static_assert(
    offsetof(struct runtime_state_fixture_cache, n_stream) == 32U,
    "fixture cache n_stream offset mismatch");
_Static_assert(
    offsetof(struct runtime_state_fixture_cache, layers) == 272U,
    "fixture cache layers offset mismatch");

uint32_t runtime_state_fixture_n_pos_per_embd(const void *hparams);

void runtime_state_fixture_kv_constructor(
    void *cache_object,
    const void *model,
    const void *hparams,
    int32_t type_k,
    int32_t type_v,
    bool v_trans,
    bool offload,
    bool unified,
    uint32_t kv_size,
    uint32_t n_seq_max);

struct runtime_state_fixture_tensor *runtime_state_fixture_view_2d(
    void *context,
    void *source,
    int64_t ne0,
    int64_t ne1,
    size_t nb1,
    size_t offset);

void *runtime_state_fixture_buffer_get_base(
    struct runtime_state_fixture_buffer *buffer);

size_t runtime_state_fixture_buffer_get_size(
    struct runtime_state_fixture_buffer *buffer);

void runtime_state_fixture_apply_ubatch(
    void *cache_object,
    const struct runtime_state_fixture_slot_info *slot,
    const struct runtime_state_fixture_ubatch *ubatch);

void runtime_state_fixture_prepare_ubatch(
    struct runtime_state_fixture_context *context,
    const struct runtime_state_fixture_ubatch *ubatch);

void *runtime_state_fixture_process_ubatch(
    struct runtime_state_fixture_context *context,
    const struct runtime_state_fixture_ubatch *ubatch,
    struct runtime_state_fixture_memory_context *memory,
    int *status);

uint8_t runtime_state_fixture_oracle_byte(
    uint32_t stream,
    uint32_t layer,
    uint32_t cell,
    bool value_cache);

void runtime_state_fixture_pause_next_apply(void);
void runtime_state_fixture_release_apply(void);
bool runtime_state_fixture_apply_is_paused(void);

struct runtime_state_fixture_cache *runtime_state_fixture_preloaded_cache(void);

void runtime_state_fixture_cache_shift(void);
void runtime_state_fixture_context_shift(void);
void runtime_state_fixture_kv_defragmentation(void);
void runtime_state_fixture_multimodal_position(void);
void runtime_state_fixture_sequence_copy(void);
void runtime_state_fixture_sequence_remove(void);
void runtime_state_fixture_speculative_context(void);

#endif
