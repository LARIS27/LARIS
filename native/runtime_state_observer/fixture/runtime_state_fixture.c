#include "runtime_state_fixture.h"

#include <sched.h>
#include <stdatomic.h>
#include <string.h>

#define TENSOR_BYTES \
    (RUNTIME_STATE_FIXTURE_KV_SIZE * RUNTIME_STATE_FIXTURE_ROW_SIZE)
#define BACKING_BYTES \
    (RUNTIME_STATE_FIXTURE_N_LAYER * RUNTIME_STATE_FIXTURE_N_STREAM * TENSOR_BYTES)

static _Alignas(64) uint8_t key_backing[BACKING_BYTES];
static _Alignas(64) uint8_t value_backing[BACKING_BYTES];
static struct runtime_state_fixture_tensor tensors[RUNTIME_STATE_FIXTURE_TENSOR_COUNT];
static struct runtime_state_fixture_tensor source_tensors[
    2U * RUNTIME_STATE_FIXTURE_N_LAYER];
static struct runtime_state_fixture_tensor preloaded_tensors[
    RUNTIME_STATE_FIXTURE_TENSOR_COUNT];
static struct runtime_state_fixture_tensor *preloaded_k_streams[
    RUNTIME_STATE_FIXTURE_N_LAYER][RUNTIME_STATE_FIXTURE_N_STREAM];
static struct runtime_state_fixture_tensor *preloaded_v_streams[
    RUNTIME_STATE_FIXTURE_N_LAYER][RUNTIME_STATE_FIXTURE_N_STREAM];
static struct runtime_state_fixture_cache_layer preloaded_layers[
    RUNTIME_STATE_FIXTURE_N_LAYER];
static uint8_t preloaded_hparams[20940U];
static struct runtime_state_fixture_cache preloaded_cache;
static struct runtime_state_fixture_buffer key_buffer = {
    key_backing,
    sizeof(key_backing),
};
static struct runtime_state_fixture_buffer value_buffer = {
    value_backing,
    sizeof(value_backing),
};
static uint32_t tensor_cursor;
static _Atomic uint64_t fixture_checksum;
static _Atomic bool pause_next_apply;
static _Atomic bool apply_is_paused;

static void prepare_preloaded_cache(void) {
    uint32_t layer;
    uint32_t stream;
    preloaded_cache.hparams = preloaded_hparams;
    preloaded_cache.v_trans = true;
    preloaded_cache.n_seq_max = RUNTIME_STATE_FIXTURE_N_STREAM;
    preloaded_cache.n_stream = RUNTIME_STATE_FIXTURE_N_STREAM;
    preloaded_cache.layers.begin = preloaded_layers;
    preloaded_cache.layers.end = preloaded_layers + RUNTIME_STATE_FIXTURE_N_LAYER;
    preloaded_cache.layers.capacity_end = preloaded_cache.layers.end;
    for (layer = 0U; layer < RUNTIME_STATE_FIXTURE_N_LAYER; ++layer) {
        struct runtime_state_fixture_cache_layer *cache_layer =
            &preloaded_layers[layer];
        cache_layer->il = layer;
        cache_layer->k_stream.begin = preloaded_k_streams[layer];
        cache_layer->k_stream.end =
            preloaded_k_streams[layer] + RUNTIME_STATE_FIXTURE_N_STREAM;
        cache_layer->k_stream.capacity_end = cache_layer->k_stream.end;
        cache_layer->v_stream.begin = preloaded_v_streams[layer];
        cache_layer->v_stream.end =
            preloaded_v_streams[layer] + RUNTIME_STATE_FIXTURE_N_STREAM;
        cache_layer->v_stream.capacity_end = cache_layer->v_stream.end;
        for (stream = 0U; stream < RUNTIME_STATE_FIXTURE_N_STREAM; ++stream) {
            const uint32_t key_ordinal =
                2U * (layer * RUNTIME_STATE_FIXTURE_N_STREAM + stream);
            const uint32_t value_ordinal = key_ordinal + 1U;
            struct runtime_state_fixture_tensor *key =
                &preloaded_tensors[key_ordinal];
            struct runtime_state_fixture_tensor *value =
                &preloaded_tensors[value_ordinal];
            key->type = 1;
            key->ne[0] = RUNTIME_STATE_FIXTURE_EMBEDDING_WIDTH;
            key->ne[1] = RUNTIME_STATE_FIXTURE_KV_SIZE;
            key->nb[1] = RUNTIME_STATE_FIXTURE_ROW_SIZE;
            key->data = key_backing
                + (layer * RUNTIME_STATE_FIXTURE_N_STREAM + stream)
                    * TENSOR_BYTES;
            value->type = 1;
            value->ne[0] = RUNTIME_STATE_FIXTURE_EMBEDDING_WIDTH;
            value->ne[1] = RUNTIME_STATE_FIXTURE_KV_SIZE;
            value->nb[1] = RUNTIME_STATE_FIXTURE_ROW_SIZE;
            value->data = value_backing
                + (layer * RUNTIME_STATE_FIXTURE_N_STREAM + stream)
                    * TENSOR_BYTES;
            preloaded_k_streams[layer][stream] = key;
            preloaded_v_streams[layer][stream] = value;
        }
    }
}

struct runtime_state_fixture_cache *runtime_state_fixture_preloaded_cache(void) {
    static bool prepared;
    if (!prepared) {
        prepare_preloaded_cache();
        prepared = true;
    }
    return &preloaded_cache;
}

__attribute__((noinline, visibility("default")))
uint32_t runtime_state_fixture_n_pos_per_embd(const void *hparams) {
    return hparams == NULL ? 0U : 1U;
}

__attribute__((noinline, visibility("default")))
struct runtime_state_fixture_tensor *runtime_state_fixture_view_2d(
    void *context,
    void *source,
    int64_t ne0,
    int64_t ne1,
    size_t nb1,
    size_t offset
) {
    struct runtime_state_fixture_tensor *tensor;
    struct runtime_state_fixture_tensor *source_tensor = source;
    (void)context;
    if (tensor_cursor >= RUNTIME_STATE_FIXTURE_TENSOR_COUNT
        || source_tensor == NULL || source_tensor->data == NULL
        || ne0 <= 0 || ne1 <= 0 || nb1 == 0U) {
        return NULL;
    }
    tensor = &tensors[tensor_cursor];
    tensor->type = 1;
    tensor->data = (tensor_cursor % 2U) == 0U
        ? (uint8_t *)source_tensor->data + offset
        : NULL;
    tensor_cursor += 1U;
    (void)atomic_fetch_add_explicit(
        &fixture_checksum,
        (uint64_t)ne0 + (uint64_t)ne1 + (uint64_t)nb1,
        memory_order_relaxed);
    return tensor;
}

__attribute__((noinline, visibility("default")))
void *runtime_state_fixture_buffer_get_base(
    struct runtime_state_fixture_buffer *buffer
) {
    return buffer == NULL ? NULL : buffer->base;
}

__attribute__((noinline, visibility("default")))
size_t runtime_state_fixture_buffer_get_size(
    struct runtime_state_fixture_buffer *buffer
) {
    return buffer == NULL ? 0U : buffer->size;
}

__attribute__((noinline, visibility("default")))
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
    uint32_t n_seq_max
) {
    uint32_t layer;
    uint32_t stream;
    tensor_cursor = 0U;
    (void)cache_object;
    (void)model;
    (void)hparams;
    (void)type_k;
    (void)type_v;
    (void)v_trans;
    (void)offload;
    (void)unified;
    (void)n_seq_max;
    (void)runtime_state_fixture_buffer_get_base(&key_buffer);
    (void)runtime_state_fixture_buffer_get_size(&key_buffer);
    (void)runtime_state_fixture_buffer_get_base(&value_buffer);
    (void)runtime_state_fixture_buffer_get_size(&value_buffer);
    for (layer = 0U; layer < RUNTIME_STATE_FIXTURE_N_LAYER; ++layer) {
        uint8_t *key_layer = key_backing
            + layer * RUNTIME_STATE_FIXTURE_N_STREAM * TENSOR_BYTES;
        uint8_t *value_layer = value_backing
            + layer * RUNTIME_STATE_FIXTURE_N_STREAM * TENSOR_BYTES;
        struct runtime_state_fixture_tensor *key_source =
            &source_tensors[2U * layer];
        struct runtime_state_fixture_tensor *value_source =
            &source_tensors[2U * layer + 1U];
        key_source->type = 1;
        key_source->data = key_layer;
        value_source->type = 1;
        value_source->data = value_layer;
        for (stream = 0U; stream < RUNTIME_STATE_FIXTURE_N_STREAM; ++stream) {
            (void)runtime_state_fixture_view_2d(
                NULL,
                key_source,
                RUNTIME_STATE_FIXTURE_EMBEDDING_WIDTH,
                kv_size,
                RUNTIME_STATE_FIXTURE_ROW_SIZE,
                stream * TENSOR_BYTES);
            (void)runtime_state_fixture_view_2d(
                NULL,
                value_source,
                RUNTIME_STATE_FIXTURE_EMBEDDING_WIDTH,
                kv_size,
                RUNTIME_STATE_FIXTURE_ROW_SIZE,
                stream * TENSOR_BYTES);
        }
    }
}

__attribute__((noinline, visibility("default")))
void runtime_state_fixture_apply_ubatch(
    void *cache_object,
    const struct runtime_state_fixture_slot_info *slot,
    const struct runtime_state_fixture_ubatch *ubatch
) {
    uintptr_t value = (uintptr_t)cache_object;
    if (atomic_load_explicit(&pause_next_apply, memory_order_acquire)) {
        atomic_store_explicit(&apply_is_paused, true, memory_order_release);
        while (atomic_load_explicit(&pause_next_apply, memory_order_acquire)) {
            sched_yield();
        }
        atomic_store_explicit(&apply_is_paused, false, memory_order_release);
    }
    if (slot != NULL) {
        value += slot->s0 + slot->s1;
    }
    if (ubatch != NULL) {
        value += ubatch->n_tokens;
    }
    (void)atomic_fetch_add_explicit(
        &fixture_checksum, (uint64_t)value, memory_order_relaxed);
}

static void write_fixture_kv_rows(
    const struct runtime_state_fixture_context *context,
    uint32_t write_generation
) {
    const struct runtime_state_fixture_slot_info *slot = context->slot;
    const uint32_t *streams;
    const struct runtime_state_fixture_vector *cell_vectors;
    size_t stream_count;
    size_t cell_vector_count;
    size_t stream_ordinal;
    uint32_t layer;

    if (slot == NULL || slot->strm.begin == NULL || slot->strm.end == NULL
        || slot->idxs.begin == NULL || slot->idxs.end == NULL
        || (const uint32_t *)slot->strm.end
            < (const uint32_t *)slot->strm.begin
        || (const struct runtime_state_fixture_vector *)slot->idxs.end
            < (const struct runtime_state_fixture_vector *)slot->idxs.begin) {
        return;
    }
    stream_count = (
        (const uint32_t *)slot->strm.end
        - (const uint32_t *)slot->strm.begin);
    cell_vector_count = (size_t)(
        (const struct runtime_state_fixture_vector *)slot->idxs.end
        - (const struct runtime_state_fixture_vector *)slot->idxs.begin);
    if (stream_count != cell_vector_count) {
        return;
    }
    streams = slot->strm.begin;
    cell_vectors = slot->idxs.begin;
    for (stream_ordinal = 0U;
         stream_ordinal < stream_count;
         ++stream_ordinal) {
        const uint32_t stream = streams[stream_ordinal];
        const struct runtime_state_fixture_vector *cells =
            &cell_vectors[stream_ordinal];
        const uint32_t *cell;
        if (stream >= RUNTIME_STATE_FIXTURE_N_STREAM
            || cells->begin == NULL || cells->end == NULL
            || (const uint32_t *)cells->end
                < (const uint32_t *)cells->begin) {
            continue;
        }
        for (layer = 0U; layer < RUNTIME_STATE_FIXTURE_N_LAYER; ++layer) {
            struct runtime_state_fixture_tensor *key =
                preloaded_k_streams[layer][stream];
            struct runtime_state_fixture_tensor *value =
                preloaded_v_streams[layer][stream];
            for (cell = cells->begin; cell < (const uint32_t *)cells->end; ++cell) {
                uint32_t element;
                if (*cell >= RUNTIME_STATE_FIXTURE_KV_SIZE) {
                    continue;
                }
                (void)memset(
                    (uint8_t *)key->data + *cell * RUNTIME_STATE_FIXTURE_ROW_SIZE,
                    (int)(0x40U + write_generation),
                    RUNTIME_STATE_FIXTURE_ROW_SIZE);
                for (element = 0U;
                     element < RUNTIME_STATE_FIXTURE_EMBEDDING_WIDTH;
                     ++element) {
                    (void)memset(
                        (uint8_t *)value->data
                            + (element * RUNTIME_STATE_FIXTURE_KV_SIZE + *cell)
                                * RUNTIME_STATE_FIXTURE_ELEMENT_SIZE,
                        (int)(0x80U + write_generation),
                        RUNTIME_STATE_FIXTURE_ELEMENT_SIZE);
                }
            }
        }
    }
}

__attribute__((noinline, visibility("default")))
void runtime_state_fixture_prepare_ubatch(
    struct runtime_state_fixture_context *context,
    const struct runtime_state_fixture_ubatch *ubatch
) {
    if (context == NULL) {
        return;
    }
    runtime_state_fixture_apply_ubatch(
        context->cache_object, context->slot, ubatch);
}

__attribute__((noinline, visibility("default")))
void *runtime_state_fixture_process_ubatch(
    struct runtime_state_fixture_context *context,
    const struct runtime_state_fixture_ubatch *ubatch,
    struct runtime_state_fixture_memory_context *memory,
    int *status
) {
    if (context == NULL || memory == NULL || status == NULL) {
        return NULL;
    }
    runtime_state_fixture_apply_ubatch(
        context->cache_object, context->slot, ubatch);
    if (memory->graph_status != 0) {
        *status = memory->graph_status;
        return NULL;
    }
    write_fixture_kv_rows(context, memory->write_generation);
    *status = 0;
    return context;
}

uint8_t runtime_state_fixture_oracle_byte(
    uint32_t stream,
    uint32_t layer,
    uint32_t cell,
    bool value_cache
) {
    struct runtime_state_fixture_tensor *tensor;
    (void)runtime_state_fixture_preloaded_cache();
    if (stream >= RUNTIME_STATE_FIXTURE_N_STREAM
        || layer >= RUNTIME_STATE_FIXTURE_N_LAYER
        || cell >= RUNTIME_STATE_FIXTURE_KV_SIZE) {
        return 0U;
    }
    tensor = value_cache
        ? preloaded_v_streams[layer][stream]
        : preloaded_k_streams[layer][stream];
    return ((const uint8_t *)tensor->data)[
        value_cache
            ? cell * RUNTIME_STATE_FIXTURE_ELEMENT_SIZE
            : cell * RUNTIME_STATE_FIXTURE_ROW_SIZE];
}

void runtime_state_fixture_pause_next_apply(void) {
    atomic_store_explicit(&pause_next_apply, true, memory_order_release);
}

void runtime_state_fixture_release_apply(void) {
    atomic_store_explicit(&pause_next_apply, false, memory_order_release);
}

bool runtime_state_fixture_apply_is_paused(void) {
    return atomic_load_explicit(&apply_is_paused, memory_order_acquire);
}

void runtime_state_fixture_cache_shift(void) {}
void runtime_state_fixture_context_shift(void) {}
void runtime_state_fixture_kv_defragmentation(void) {}
void runtime_state_fixture_multimodal_position(void) {}
void runtime_state_fixture_sequence_copy(void) {}
void runtime_state_fixture_sequence_remove(void) {}
void runtime_state_fixture_speculative_context(void) {}
