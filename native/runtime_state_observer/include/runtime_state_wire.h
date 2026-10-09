#ifndef LARIS_RUNTIME_STATE_WIRE_H
#define LARIS_RUNTIME_STATE_WIRE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define RUNTIME_STATE_WIRE_MAGIC UINT32_C(0x5352534c)
#define RUNTIME_STATE_WIRE_VERSION UINT16_C(5)

#define RUNTIME_STATE_FRAME_TOPOLOGY UINT16_C(1)
#define RUNTIME_STATE_FRAME_TENSOR_LAYOUT UINT16_C(2)
#define RUNTIME_STATE_FRAME_KV_BACKING UINT16_C(3)
#define RUNTIME_STATE_FRAME_TOKEN_CELL UINT16_C(4)
#define RUNTIME_STATE_FRAME_APPLY_COMPLETE UINT16_C(5)
#define RUNTIME_STATE_FRAME_WATERMARK UINT16_C(6)
#define RUNTIME_STATE_FRAME_TERMINAL UINT16_C(7)
#define RUNTIME_STATE_FRAME_MUTATION UINT16_C(8)
#define RUNTIME_STATE_FRAME_STREAM_VERSION UINT16_C(9)
#define RUNTIME_STATE_FRAME_STATE_UNIT_FRAGMENT UINT16_C(10)
#define RUNTIME_STATE_FRAME_STATE_UNIT_SEAL UINT16_C(11)
#define RUNTIME_STATE_FRAME_TOKEN_CELL_RANGE UINT16_C(12)
#define RUNTIME_STATE_FRAME_STATE_UNIT_FRAGMENT_BATCH UINT16_C(13)
#define RUNTIME_STATE_FRAME_STATE_UNIT_SCHEDULE_BATCH UINT16_C(14)
#define RUNTIME_STATE_FRAME_STATE_UNIT_BULK UINT16_C(15)

#define RUNTIME_STATE_CHANNEL_METADATA UINT32_C(0)
#define RUNTIME_STATE_CHANNEL_STATE_UNIT UINT32_C(1)
#define RUNTIME_STATE_CHANNEL_COUNT ((size_t)2)
#ifndef RUNTIME_STATE_MAX_FRAGMENT_BYTES
#define RUNTIME_STATE_MAX_FRAGMENT_BYTES 4096U
#endif

#define RUNTIME_STATE_HEADER_SIZE ((size_t)24)
#define RUNTIME_STATE_TOPOLOGY_SIZE ((size_t)48)
#define RUNTIME_STATE_TENSOR_LAYOUT_SIZE ((size_t)80)
#define RUNTIME_STATE_KV_BACKING_SIZE ((size_t)40)
#define RUNTIME_STATE_TOKEN_CELL_SIZE ((size_t)48)
#define RUNTIME_STATE_TOKEN_CELL_RANGE_SIZE ((size_t)56)
#define RUNTIME_STATE_APPLY_COMPLETE_SIZE ((size_t)24)
#define RUNTIME_STATE_MUTATION_SIZE ((size_t)32)
#define RUNTIME_STATE_STREAM_VERSION_SIZE ((size_t)136)
#define RUNTIME_STATE_COUNTERS_SIZE ((size_t)136)
#define RUNTIME_STATE_STATE_UNIT_FRAGMENT_METADATA_SIZE ((size_t)72)
#define RUNTIME_STATE_STATE_UNIT_FRAGMENT_BATCH_PREFIX_SIZE ((size_t)8)
#define RUNTIME_STATE_STATE_UNIT_SCHEDULE_BATCH_PREFIX_SIZE ((size_t)96)
#define RUNTIME_STATE_STATE_UNIT_SEAL_SIZE ((size_t)80)
#define RUNTIME_STATE_STATE_UNIT_BULK_PREFIX_SIZE ((size_t)128)
#define RUNTIME_STATE_BULK_PAYLOAD_BYTES ((size_t)65536)

#define RUNTIME_STATE_TOPOLOGY_FRAME_SIZE (RUNTIME_STATE_HEADER_SIZE + RUNTIME_STATE_TOPOLOGY_SIZE)
#define RUNTIME_STATE_TENSOR_FRAME_SIZE (RUNTIME_STATE_HEADER_SIZE + RUNTIME_STATE_TENSOR_LAYOUT_SIZE)
#define RUNTIME_STATE_BACKING_FRAME_SIZE (RUNTIME_STATE_HEADER_SIZE + RUNTIME_STATE_KV_BACKING_SIZE)
#define RUNTIME_STATE_TOKEN_FRAME_SIZE (RUNTIME_STATE_HEADER_SIZE + RUNTIME_STATE_TOKEN_CELL_SIZE)
#define RUNTIME_STATE_TOKEN_CELL_RANGE_FRAME_SIZE \
    (RUNTIME_STATE_HEADER_SIZE + RUNTIME_STATE_TOKEN_CELL_RANGE_SIZE)
#define RUNTIME_STATE_APPLY_COMPLETE_FRAME_SIZE (RUNTIME_STATE_HEADER_SIZE + RUNTIME_STATE_APPLY_COMPLETE_SIZE)
#define RUNTIME_STATE_MUTATION_FRAME_SIZE (RUNTIME_STATE_HEADER_SIZE + RUNTIME_STATE_MUTATION_SIZE)
#define RUNTIME_STATE_STREAM_VERSION_FRAME_SIZE (RUNTIME_STATE_HEADER_SIZE + RUNTIME_STATE_STREAM_VERSION_SIZE)
#define RUNTIME_STATE_COUNTERS_FRAME_SIZE (RUNTIME_STATE_HEADER_SIZE + RUNTIME_STATE_COUNTERS_SIZE)
#define RUNTIME_STATE_STATE_UNIT_SEAL_FRAME_SIZE (RUNTIME_STATE_HEADER_SIZE + RUNTIME_STATE_STATE_UNIT_SEAL_SIZE)
#define RUNTIME_STATE_STATE_UNIT_FRAGMENT_FRAME_SIZE_MAX \
    (RUNTIME_STATE_HEADER_SIZE + RUNTIME_STATE_STATE_UNIT_FRAGMENT_METADATA_SIZE \
        + RUNTIME_STATE_MAX_FRAGMENT_BYTES)
#define RUNTIME_STATE_STATE_UNIT_FRAGMENT_BATCH_FRAME_SIZE_MAX \
    (RUNTIME_STATE_HEADER_SIZE \
        + RUNTIME_STATE_STATE_UNIT_FRAGMENT_BATCH_PREFIX_SIZE \
        + RUNTIME_STATE_STATE_UNIT_FRAGMENT_METADATA_SIZE \
        + RUNTIME_STATE_MAX_FRAGMENT_BYTES)
#define RUNTIME_STATE_STATE_UNIT_BULK_FRAME_SIZE_MAX \
    (RUNTIME_STATE_HEADER_SIZE + RUNTIME_STATE_STATE_UNIT_BULK_PREFIX_SIZE \
        + RUNTIME_STATE_BULK_PAYLOAD_BYTES)
#define RUNTIME_STATE_MAX_FRAME_SIZE \
    RUNTIME_STATE_STATE_UNIT_BULK_FRAME_SIZE_MAX

#ifndef RUNTIME_STATE_TENSOR_KIND_K
#define RUNTIME_STATE_TENSOR_KIND_K UINT32_C(1)
#define RUNTIME_STATE_TENSOR_KIND_V UINT32_C(2)
#endif

#ifndef RUNTIME_STATE_MUTATION_CACHE_SHIFT
#define RUNTIME_STATE_MUTATION_CACHE_SHIFT UINT32_C(1)
#define RUNTIME_STATE_MUTATION_CONTEXT_SHIFT UINT32_C(2)
#define RUNTIME_STATE_MUTATION_KV_DEFRAGMENTATION UINT32_C(3)
#define RUNTIME_STATE_MUTATION_MULTIMODAL_POSITION UINT32_C(4)
#define RUNTIME_STATE_MUTATION_SEQUENCE_COPY UINT32_C(5)
#define RUNTIME_STATE_MUTATION_SEQUENCE_REMOVE UINT32_C(6)
#define RUNTIME_STATE_MUTATION_SEQUENCE_STREAM_REMAP UINT32_C(7)
#define RUNTIME_STATE_MUTATION_SPECULATIVE_CONTEXT UINT32_C(8)
#endif

#define RUNTIME_STATE_STREAM_EMPTY UINT32_C(0)
#define RUNTIME_STATE_STREAM_BUSY UINT32_C(1)
#define RUNTIME_STATE_STREAM_READY UINT32_C(2)
#define RUNTIME_STATE_VERSION_STREAM_COUNT ((size_t)8)

struct runtime_state_frame_header {
    uint32_t magic;
    uint16_t version;
    uint16_t frame_type;
    uint32_t payload_length;
    uint32_t channel_id;
    uint64_t frame_sequence;
};

struct runtime_state_topology_event {
    uint64_t event_id;
    uint32_t tgid;
    uint32_t tid;
    uint64_t monotonic_ns;
    uint32_t n_ctx;
    uint32_t n_stream;
    uint32_t n_seq_max;
    uint32_t n_layer;
    uint32_t v_trans;
    uint32_t n_pos_per_embd;
};

struct runtime_state_tensor_layout_event {
    uint64_t event_id;
    uint32_t tgid;
    uint32_t tid;
    uint64_t monotonic_ns;
    uint32_t kind;
    uint32_t layer;
    uint32_t stream_index;
    uint32_t ggml_type_id;
    uint64_t tensor_address;
    uint64_t row_size;
    uint64_t cell_capacity;
    uint64_t element_size;
    uint64_t embedding_width;
};

struct runtime_state_kv_backing_event {
    uint64_t event_id;
    uint32_t tgid;
    uint32_t tid;
    uint64_t monotonic_ns;
    uint64_t base_address;
    uint64_t size_bytes;
};

struct runtime_state_token_cell_event {
    uint64_t apply_call_id;
    uint32_t tgid;
    uint32_t tid;
    uint64_t monotonic_ns;
    uint32_t seq_id;
    uint32_t stream_index;
    uint64_t token_position;
    uint64_t cell_index;
};

struct runtime_state_token_cell_range_event {
    uint64_t apply_call_id;
    uint32_t tgid;
    uint32_t tid;
    uint64_t monotonic_ns;
    uint32_t seq_id;
    uint32_t stream_index;
    uint64_t first_token_position;
    uint64_t first_cell_index;
    uint32_t token_count;
    uint32_t reserved;
};

struct runtime_state_apply_complete_event {
    uint64_t apply_call_id;
    uint32_t tgid;
    uint32_t tid;
    uint64_t monotonic_ns;
};

struct runtime_state_mutation_event {
    uint64_t event_id;
    uint32_t tgid;
    uint32_t tid;
    uint64_t monotonic_ns;
    uint32_t mutation_kind;
    uint32_t reserved;
};

struct runtime_state_state_unit_fragment {
    uint64_t apply_call_id;
    uint32_t fragment_index;
    uint32_t seq_id;
    uint32_t stream_index;
    uint32_t tensor_kind;
    uint64_t token_position;
    uint64_t cell_index;
    uint32_t layer;
    uint32_t payload_size;
    uint64_t tensor_byte_offset;
    uint64_t writer_monotonic_ns;
    uint64_t span_cell_count;
};

struct runtime_state_state_unit_schedule_batch {
    uint32_t record_count;
    uint32_t payload_size;
    uint64_t apply_call_id;
    uint32_t first_fragment_index;
    uint32_t first_flat_index;
    uint32_t seq_id;
    uint32_t stream_index;
    uint32_t tensor_kind;
    uint32_t fragments_per_tensor;
    uint64_t first_token_position;
    uint64_t first_cell_index;
    uint32_t chunks_per_dimension;
    uint32_t token_count;
    uint32_t row_size;
    uint32_t cell_capacity;
    uint32_t element_size;
    uint32_t embedding_width;
    uint32_t transposed_value;
    uint32_t reserved;
    uint64_t writer_monotonic_ns;
};

struct runtime_state_state_unit_read_plan {
    uint64_t apply_call_id;
    uint64_t tensor_address;
    uint64_t first_token_position;
    uint64_t first_cell_index;
    uint64_t writer_monotonic_ns;
    uint64_t copy_byte_count;
    uint32_t first_fragment_index;
    uint32_t fragment_count;
    uint32_t seq_id;
    uint32_t stream_index;
    uint32_t tensor_kind;
    uint32_t layer;
    uint32_t expected_n_layer;
    uint32_t token_count;
    uint32_t row_size;
    uint32_t cell_capacity;
    uint32_t element_size;
    uint32_t embedding_width;
    uint32_t chunks_per_dimension;
    uint32_t fragments_per_tensor;
    uint32_t transposed_value;
    uint32_t reserved;
};

#ifndef RUNTIME_STATE_WRITER_COMPLETE
#define RUNTIME_STATE_WRITER_COMPLETE UINT32_C(1)
#endif

struct runtime_state_state_unit_seal {
    uint64_t apply_call_id;
    uint32_t expected_fragment_count;
    uint32_t expected_token_cell_count;
    uint32_t stream_mask;
    uint32_t final_writer_state;
    uint64_t layout_generation;
    uint64_t backing_generation;
    uint64_t mutation_generation;
    uint64_t copy_byte_count;
    uint64_t reserve_failure_baseline;
    uint64_t reserve_failure_final;
    uint64_t writer_return_monotonic_ns;
};

typedef struct runtime_state_state_unit_seal
    runtime_state_state_unit_apply_ready;

struct runtime_state_stream_version_snapshot {
    uint64_t snapshot_id;
    uint64_t mutation_generation;
    uint64_t backing_generation;
    uint64_t layout_generation;
    uint32_t active_stream_count;
    uint32_t reserved;
    uint64_t stream_apply_call_ids[RUNTIME_STATE_VERSION_STREAM_COUNT];
    uint32_t stream_states[RUNTIME_STATE_VERSION_STREAM_COUNT];
};

struct runtime_state_terminal_counters {
    uint64_t cutoff_monotonic_ns;
    uint64_t topology_event_count;
    uint64_t tensor_layout_event_count;
    uint64_t kv_backing_event_count;
    uint64_t token_cell_event_count;
    uint64_t apply_complete_event_count;
    uint64_t kernel_event_submit_count;
    uint64_t native_event_receive_count;
    uint64_t native_wire_frame_count;
    uint64_t ringbuf_reserve_fail_count;
    uint64_t user_memory_read_failure_count;
    uint64_t unmatched_call_count;
    uint64_t unsupported_mutation_count;
    uint64_t mapping_mutation_count;
    uint64_t final_inflight_count;
    uint64_t mutation_event_count;
    uint64_t active_emitter_count;
};

_Static_assert(sizeof(struct runtime_state_frame_header) == 24U, "wire header must be 24 bytes");
_Static_assert(sizeof(struct runtime_state_state_unit_fragment) == 72U, "state-unit fragment metadata must be 72 bytes");
_Static_assert(sizeof(struct runtime_state_state_unit_schedule_batch) == 96U, "state-unit schedule batch prefix must be 96 bytes");
_Static_assert(sizeof(struct runtime_state_state_unit_read_plan) == 112U, "state-unit read plan must be 112 bytes");
_Static_assert(sizeof(struct runtime_state_state_unit_seal) == 80U, "state-unit seal must be 80 bytes");
_Static_assert(sizeof(struct runtime_state_topology_event) == 48U, "topology event must be 48 bytes");
_Static_assert(sizeof(struct runtime_state_tensor_layout_event) == 80U, "tensor event must be 80 bytes");
_Static_assert(sizeof(struct runtime_state_kv_backing_event) == 40U, "backing event must be 40 bytes");
_Static_assert(sizeof(struct runtime_state_token_cell_event) == 48U, "token event must be 48 bytes");
_Static_assert(sizeof(struct runtime_state_token_cell_range_event) == 56U, "token range event must be 56 bytes");
_Static_assert(sizeof(struct runtime_state_apply_complete_event) == 24U, "apply completion must be 24 bytes");
_Static_assert(sizeof(struct runtime_state_mutation_event) == 32U, "mutation event must be 32 bytes");
_Static_assert(sizeof(struct runtime_state_stream_version_snapshot) == 136U, "stream version must be 136 bytes");
_Static_assert(sizeof(struct runtime_state_terminal_counters) == 136U, "counter frame must be 136 bytes");

static inline bool runtime_state_valid_state_unit_read_plan(
    const struct runtime_state_state_unit_read_plan *plan
) {
    uint64_t row_size;
    uint64_t fragments_per_tensor;
    uint64_t fragment_count;
    uint64_t copy_byte_count;
    if (plan == NULL || plan->apply_call_id == 0U
        || plan->tensor_address == 0U || plan->writer_monotonic_ns == 0U
        || plan->copy_byte_count == 0U || plan->fragment_count == 0U
        || plan->stream_index >= 8U
        || (plan->tensor_kind != RUNTIME_STATE_TENSOR_KIND_K
            && plan->tensor_kind != RUNTIME_STATE_TENSOR_KIND_V)
        || plan->expected_n_layer == 0U
        || plan->layer >= plan->expected_n_layer
        || plan->token_count == 0U || plan->row_size == 0U
        || plan->cell_capacity == 0U || plan->element_size == 0U
        || plan->embedding_width == 0U
        || plan->chunks_per_dimension == 0U
        || plan->fragments_per_tensor == 0U || plan->reserved != 0U
        || plan->first_cell_index >= plan->cell_capacity
        || plan->token_count
            > plan->cell_capacity - plan->first_cell_index
        || plan->first_fragment_index
            > UINT32_MAX - plan->fragment_count
        || plan->first_token_position > UINT64_MAX - plan->token_count
        || (plan->tensor_kind == RUNTIME_STATE_TENSOR_KIND_K
            && plan->transposed_value != 0U)
        || (plan->tensor_kind == RUNTIME_STATE_TENSOR_KIND_V
            && plan->transposed_value > 1U)) {
        return false;
    }
    row_size = (uint64_t)plan->element_size * plan->embedding_width;
    fragments_per_tensor = plan->chunks_per_dimension;
    if (plan->transposed_value != 0U) {
        fragments_per_tensor *= plan->embedding_width;
    }
    fragment_count = fragments_per_tensor;
    copy_byte_count = (uint64_t)plan->token_count * plan->row_size;
    return row_size == plan->row_size
        && fragments_per_tensor == plan->fragments_per_tensor
        && fragment_count == plan->fragment_count
        && copy_byte_count == plan->copy_byte_count;
}

static inline bool runtime_state_state_unit_read_plan_follows(
    const struct runtime_state_state_unit_read_plan *previous,
    const struct runtime_state_state_unit_read_plan *current
) {
    return runtime_state_valid_state_unit_read_plan(previous)
        && runtime_state_valid_state_unit_read_plan(current)
        && current->apply_call_id == previous->apply_call_id
        && current->writer_monotonic_ns == previous->writer_monotonic_ns
        && current->first_fragment_index
            == previous->first_fragment_index + previous->fragment_count;
}

static inline bool runtime_state_valid_state_unit_apply_ready(
    const runtime_state_state_unit_apply_ready *ready,
    const struct runtime_state_state_unit_read_plan *last_plan,
    uint32_t plan_count,
    uint64_t planned_copy_byte_count
) {
    return ready != NULL && last_plan != NULL && plan_count != 0U
        && runtime_state_valid_state_unit_read_plan(last_plan)
        && ready->apply_call_id == last_plan->apply_call_id
        && ready->expected_fragment_count
            == last_plan->first_fragment_index + last_plan->fragment_count
        && ready->expected_token_cell_count != 0U
        && ready->stream_mask != 0U
        && ready->final_writer_state == RUNTIME_STATE_WRITER_COMPLETE
        && ready->copy_byte_count == planned_copy_byte_count
        && ready->copy_byte_count != 0U
        && ready->reserve_failure_baseline == ready->reserve_failure_final
        && ready->writer_return_monotonic_ns
            == last_plan->writer_monotonic_ns;
}

static inline uint16_t runtime_state_load_le16(const uint8_t *source) {
    return (uint16_t)((uint16_t)source[0] | ((uint16_t)source[1] << 8U));
}

static inline uint32_t runtime_state_load_le32(const uint8_t *source) {
    return (uint32_t)source[0]
        | ((uint32_t)source[1] << 8U)
        | ((uint32_t)source[2] << 16U)
        | ((uint32_t)source[3] << 24U);
}

static inline uint64_t runtime_state_load_le64(const uint8_t *source) {
    return (uint64_t)runtime_state_load_le32(source)
        | ((uint64_t)runtime_state_load_le32(source + 4U) << 32U);
}

static inline void runtime_state_store_le16(uint8_t *destination, uint16_t value) {
    destination[0] = (uint8_t)value;
    destination[1] = (uint8_t)(value >> 8U);
}

static inline void runtime_state_store_le32(uint8_t *destination, uint32_t value) {
    destination[0] = (uint8_t)value;
    destination[1] = (uint8_t)(value >> 8U);
    destination[2] = (uint8_t)(value >> 16U);
    destination[3] = (uint8_t)(value >> 24U);
}

static inline void runtime_state_store_le64(uint8_t *destination, uint64_t value) {
    runtime_state_store_le32(destination, (uint32_t)value);
    runtime_state_store_le32(destination + 4U, (uint32_t)(value >> 32U));
}

static inline size_t runtime_state_payload_size(uint16_t frame_type) {
    switch (frame_type) {
        case RUNTIME_STATE_FRAME_TOPOLOGY: return RUNTIME_STATE_TOPOLOGY_SIZE;
        case RUNTIME_STATE_FRAME_TENSOR_LAYOUT: return RUNTIME_STATE_TENSOR_LAYOUT_SIZE;
        case RUNTIME_STATE_FRAME_KV_BACKING: return RUNTIME_STATE_KV_BACKING_SIZE;
        case RUNTIME_STATE_FRAME_TOKEN_CELL: return RUNTIME_STATE_TOKEN_CELL_SIZE;
        case RUNTIME_STATE_FRAME_TOKEN_CELL_RANGE: return RUNTIME_STATE_TOKEN_CELL_RANGE_SIZE;
        case RUNTIME_STATE_FRAME_APPLY_COMPLETE: return RUNTIME_STATE_APPLY_COMPLETE_SIZE;
        case RUNTIME_STATE_FRAME_MUTATION: return RUNTIME_STATE_MUTATION_SIZE;
        case RUNTIME_STATE_FRAME_STREAM_VERSION: return RUNTIME_STATE_STREAM_VERSION_SIZE;
        case RUNTIME_STATE_FRAME_STATE_UNIT_SEAL: return RUNTIME_STATE_STATE_UNIT_SEAL_SIZE;
        case RUNTIME_STATE_FRAME_WATERMARK: return RUNTIME_STATE_COUNTERS_SIZE;
        case RUNTIME_STATE_FRAME_TERMINAL: return RUNTIME_STATE_COUNTERS_SIZE;
        default: return 0U;
    }
}

static inline uint32_t runtime_state_frame_channel(uint16_t frame_type) {
    return frame_type == RUNTIME_STATE_FRAME_STATE_UNIT_FRAGMENT
            || frame_type == RUNTIME_STATE_FRAME_STATE_UNIT_FRAGMENT_BATCH
            || frame_type == RUNTIME_STATE_FRAME_STATE_UNIT_SCHEDULE_BATCH
            || frame_type == RUNTIME_STATE_FRAME_STATE_UNIT_BULK
            || frame_type == RUNTIME_STATE_FRAME_STATE_UNIT_SEAL
        ? RUNTIME_STATE_CHANNEL_STATE_UNIT
        : RUNTIME_STATE_CHANNEL_METADATA;
}

static inline bool runtime_state_valid_payload_length(
    uint16_t frame_type,
    uint32_t payload_length
) {
    const size_t fixed_size = runtime_state_payload_size(frame_type);
    if (frame_type == RUNTIME_STATE_FRAME_STATE_UNIT_FRAGMENT) {
        return payload_length > RUNTIME_STATE_STATE_UNIT_FRAGMENT_METADATA_SIZE
            && payload_length <= RUNTIME_STATE_STATE_UNIT_FRAGMENT_METADATA_SIZE
                + RUNTIME_STATE_MAX_FRAGMENT_BYTES;
    }
    if (frame_type == RUNTIME_STATE_FRAME_STATE_UNIT_FRAGMENT_BATCH) {
        return payload_length
                > RUNTIME_STATE_STATE_UNIT_FRAGMENT_BATCH_PREFIX_SIZE
                    + RUNTIME_STATE_STATE_UNIT_FRAGMENT_METADATA_SIZE
            && payload_length
                <= RUNTIME_STATE_STATE_UNIT_FRAGMENT_BATCH_PREFIX_SIZE
                    + RUNTIME_STATE_STATE_UNIT_FRAGMENT_METADATA_SIZE
                    + RUNTIME_STATE_MAX_FRAGMENT_BYTES;
    }
    if (frame_type == RUNTIME_STATE_FRAME_STATE_UNIT_SCHEDULE_BATCH) {
        return payload_length
                > RUNTIME_STATE_STATE_UNIT_SCHEDULE_BATCH_PREFIX_SIZE
            && payload_length
                <= RUNTIME_STATE_STATE_UNIT_SCHEDULE_BATCH_PREFIX_SIZE
                    + RUNTIME_STATE_MAX_FRAGMENT_BYTES;
    }
    if (frame_type == RUNTIME_STATE_FRAME_STATE_UNIT_BULK) {
        return payload_length > RUNTIME_STATE_STATE_UNIT_BULK_PREFIX_SIZE
            && payload_length
                <= RUNTIME_STATE_STATE_UNIT_BULK_PREFIX_SIZE
                    + RUNTIME_STATE_BULK_PAYLOAD_BYTES;
    }
    return fixed_size != 0U && payload_length == fixed_size;
}

static inline bool runtime_state_encode_header_for_channel(
    uint8_t *destination,
    size_t destination_size,
    uint16_t frame_type,
    uint32_t payload_size,
    uint32_t channel_id,
    uint64_t frame_sequence
) {
    if (destination == NULL || destination_size < RUNTIME_STATE_HEADER_SIZE
        || !runtime_state_valid_payload_length(frame_type, payload_size)
        || channel_id != runtime_state_frame_channel(frame_type)
        || frame_sequence == 0U) {
        return false;
    }
    runtime_state_store_le32(destination + 0U, RUNTIME_STATE_WIRE_MAGIC);
    runtime_state_store_le16(destination + 4U, RUNTIME_STATE_WIRE_VERSION);
    runtime_state_store_le16(destination + 6U, frame_type);
    runtime_state_store_le32(destination + 8U, payload_size);
    runtime_state_store_le32(destination + 12U, channel_id);
    runtime_state_store_le64(destination + 16U, frame_sequence);
    return true;
}

static inline bool runtime_state_encode_header(
    uint8_t *destination,
    size_t destination_size,
    uint16_t frame_type,
    uint64_t frame_sequence
) {
    const size_t payload_size = runtime_state_payload_size(frame_type);
    return payload_size <= UINT32_MAX
        && runtime_state_encode_header_for_channel(
            destination, destination_size, frame_type,
            (uint32_t)payload_size, RUNTIME_STATE_CHANNEL_METADATA,
            frame_sequence);
}

static inline bool runtime_state_decode_header(
    const uint8_t *source,
    size_t source_size,
    struct runtime_state_frame_header *header
) {
    if (source == NULL || header == NULL || source_size < RUNTIME_STATE_HEADER_SIZE) {
        return false;
    }
    header->magic = runtime_state_load_le32(source + 0U);
    header->version = runtime_state_load_le16(source + 4U);
    header->frame_type = runtime_state_load_le16(source + 6U);
    header->payload_length = runtime_state_load_le32(source + 8U);
    header->channel_id = runtime_state_load_le32(source + 12U);
    header->frame_sequence = runtime_state_load_le64(source + 16U);
    return header->magic == RUNTIME_STATE_WIRE_MAGIC
        && header->version == RUNTIME_STATE_WIRE_VERSION
        && header->channel_id == runtime_state_frame_channel(header->frame_type)
        && header->frame_sequence != 0U
        && runtime_state_valid_payload_length(
            header->frame_type, header->payload_length);
}

static inline bool runtime_state_valid_topology(const struct runtime_state_topology_event *event) {
    return event != NULL && event->event_id != 0U && event->tgid != 0U && event->tid != 0U
        && event->n_ctx != 0U && event->n_stream != 0U && event->n_seq_max != 0U
        && event->n_layer != 0U && event->v_trans <= 1U && event->n_pos_per_embd != 0U;
}

static inline bool runtime_state_valid_tensor(const struct runtime_state_tensor_layout_event *event) {
    return event != NULL && event->event_id != 0U && event->tgid != 0U && event->tid != 0U
        && (event->kind == RUNTIME_STATE_TENSOR_KIND_K || event->kind == RUNTIME_STATE_TENSOR_KIND_V)
        && event->tensor_address != 0U && event->row_size != 0U && event->cell_capacity != 0U
        && event->element_size != 0U && event->embedding_width != 0U;
}

static inline bool runtime_state_valid_backing(const struct runtime_state_kv_backing_event *event) {
    return event != NULL && event->event_id != 0U && event->tgid != 0U && event->tid != 0U
        && event->base_address != 0U && event->size_bytes != 0U;
}

static inline bool runtime_state_valid_token(const struct runtime_state_token_cell_event *event) {
    return event != NULL && event->apply_call_id != 0U && event->tgid != 0U && event->tid != 0U;
}

static inline bool runtime_state_valid_token_range(
    const struct runtime_state_token_cell_range_event *event
) {
    return event != NULL && event->apply_call_id != 0U
        && event->tgid != 0U && event->tid != 0U
        && event->stream_index < RUNTIME_STATE_VERSION_STREAM_COUNT
        && event->token_count != 0U && event->token_count <= 2048U
        && event->reserved == 0U
        && event->first_token_position
            <= UINT64_MAX - ((uint64_t)event->token_count - 1U)
        && event->first_cell_index
            <= UINT64_MAX - ((uint64_t)event->token_count - 1U);
}

static inline bool runtime_state_valid_apply_complete(
    const struct runtime_state_apply_complete_event *event
) {
    return event != NULL && event->apply_call_id != 0U
        && event->tgid != 0U && event->tid != 0U;
}

static inline bool runtime_state_valid_mutation(
    const struct runtime_state_mutation_event *event
) {
    return event != NULL && event->event_id != 0U
        && event->tgid != 0U && event->tid != 0U
        && event->mutation_kind >= RUNTIME_STATE_MUTATION_CACHE_SHIFT
        && event->mutation_kind <= RUNTIME_STATE_MUTATION_SPECULATIVE_CONTEXT
        && event->reserved == 0U;
}

static inline bool runtime_state_valid_stream_version(
    const struct runtime_state_stream_version_snapshot *snapshot
) {
    size_t index;
    if (snapshot == NULL || snapshot->snapshot_id == 0U
        || (snapshot->active_stream_count != 4U
            && snapshot->active_stream_count != 8U)
        || snapshot->reserved != 0U) {
        return false;
    }
    for (index = 0U; index < RUNTIME_STATE_VERSION_STREAM_COUNT; ++index) {
        const uint32_t state = snapshot->stream_states[index];
        const uint64_t apply_call_id = snapshot->stream_apply_call_ids[index];
        if (state > RUNTIME_STATE_STREAM_READY
            || ((state == RUNTIME_STATE_STREAM_EMPTY) != (apply_call_id == 0U))
            || (index >= snapshot->active_stream_count
                && (state != RUNTIME_STATE_STREAM_EMPTY
                    || apply_call_id != 0U))) {
            return false;
        }
    }
    return true;
}

static inline bool runtime_state_valid_state_unit_fragment(
    const struct runtime_state_state_unit_fragment *fragment,
    const uint8_t *payload
) {
    return fragment != NULL && payload != NULL
        && fragment->apply_call_id != 0U
        && fragment->stream_index < RUNTIME_STATE_VERSION_STREAM_COUNT
        && (fragment->tensor_kind == RUNTIME_STATE_TENSOR_KIND_K
            || fragment->tensor_kind == RUNTIME_STATE_TENSOR_KIND_V)
        && fragment->payload_size != 0U
        && fragment->payload_size <= RUNTIME_STATE_MAX_FRAGMENT_BYTES
        && fragment->span_cell_count != 0U;
}

static inline bool runtime_state_valid_state_unit_seal(
    const struct runtime_state_state_unit_seal *seal
) {
    return seal != NULL && seal->apply_call_id != 0U
        && seal->expected_fragment_count != 0U
        && seal->expected_token_cell_count != 0U
        && seal->stream_mask != 0U && (seal->stream_mask & ~UINT32_C(0xff)) == 0U
        && seal->final_writer_state == RUNTIME_STATE_WRITER_COMPLETE
        && seal->copy_byte_count != 0U
        && seal->reserve_failure_baseline == seal->reserve_failure_final;
}

static inline bool runtime_state_encode_state_unit_fragment_record(
    uint8_t *destination,
    size_t destination_size,
    const struct runtime_state_state_unit_fragment *fragment,
    const uint8_t *fragment_payload,
    size_t *record_size
);

static inline bool runtime_state_encode_state_unit_fragment_frame(
    uint8_t *destination,
    size_t destination_size,
    const struct runtime_state_state_unit_fragment *fragment,
    const uint8_t *fragment_payload,
    uint64_t sequence
) {
    uint8_t *payload;
    size_t record_size;
    size_t payload_size;
    if (!runtime_state_valid_state_unit_fragment(fragment, fragment_payload)) {
        return false;
    }
    payload_size = RUNTIME_STATE_STATE_UNIT_FRAGMENT_METADATA_SIZE
        + (size_t)fragment->payload_size;
    if (destination_size < RUNTIME_STATE_HEADER_SIZE + payload_size
        || !runtime_state_encode_header_for_channel(
            destination, destination_size,
            RUNTIME_STATE_FRAME_STATE_UNIT_FRAGMENT,
            (uint32_t)payload_size, RUNTIME_STATE_CHANNEL_STATE_UNIT,
            sequence)) {
        return false;
    }
    payload = destination + RUNTIME_STATE_HEADER_SIZE;
    return runtime_state_encode_state_unit_fragment_record(
        payload, payload_size, fragment, fragment_payload, &record_size)
        && record_size == payload_size;
}

static inline bool runtime_state_encode_state_unit_fragment_record(
    uint8_t *destination,
    size_t destination_size,
    const struct runtime_state_state_unit_fragment *fragment,
    const uint8_t *fragment_payload,
    size_t *record_size
) {
    size_t encoded_size;
    if (record_size == NULL
        || !runtime_state_valid_state_unit_fragment(
            fragment, fragment_payload)) {
        return false;
    }
    encoded_size = RUNTIME_STATE_STATE_UNIT_FRAGMENT_METADATA_SIZE
        + (size_t)fragment->payload_size;
    if (destination == NULL || destination_size < encoded_size) {
        return false;
    }
    *record_size = encoded_size;
    runtime_state_store_le64(destination + 0U, fragment->apply_call_id);
    runtime_state_store_le32(destination + 8U, fragment->fragment_index);
    runtime_state_store_le32(destination + 12U, fragment->seq_id);
    runtime_state_store_le32(destination + 16U, fragment->stream_index);
    runtime_state_store_le32(destination + 20U, fragment->tensor_kind);
    runtime_state_store_le64(destination + 24U, fragment->token_position);
    runtime_state_store_le64(destination + 32U, fragment->cell_index);
    runtime_state_store_le32(destination + 40U, fragment->layer);
    runtime_state_store_le32(destination + 44U, fragment->payload_size);
    runtime_state_store_le64(destination + 48U, fragment->tensor_byte_offset);
    runtime_state_store_le64(destination + 56U, fragment->writer_monotonic_ns);
    runtime_state_store_le64(destination + 64U, fragment->span_cell_count);
    memcpy(
        destination + RUNTIME_STATE_STATE_UNIT_FRAGMENT_METADATA_SIZE,
        fragment_payload,
        fragment->payload_size);
    return true;
}

static inline bool runtime_state_decode_state_unit_fragment_record(
    const uint8_t *source,
    size_t source_size,
    struct runtime_state_state_unit_fragment *fragment,
    const uint8_t **fragment_payload,
    size_t *record_size
) {
    size_t decoded_size;
    if (source == NULL || fragment == NULL || fragment_payload == NULL
        || record_size == NULL
        || source_size <= RUNTIME_STATE_STATE_UNIT_FRAGMENT_METADATA_SIZE) {
        return false;
    }
    fragment->apply_call_id = runtime_state_load_le64(source + 0U);
    fragment->fragment_index = runtime_state_load_le32(source + 8U);
    fragment->seq_id = runtime_state_load_le32(source + 12U);
    fragment->stream_index = runtime_state_load_le32(source + 16U);
    fragment->tensor_kind = runtime_state_load_le32(source + 20U);
    fragment->token_position = runtime_state_load_le64(source + 24U);
    fragment->cell_index = runtime_state_load_le64(source + 32U);
    fragment->layer = runtime_state_load_le32(source + 40U);
    fragment->payload_size = runtime_state_load_le32(source + 44U);
    fragment->tensor_byte_offset = runtime_state_load_le64(source + 48U);
    fragment->writer_monotonic_ns = runtime_state_load_le64(source + 56U);
    fragment->span_cell_count = runtime_state_load_le64(source + 64U);
    decoded_size = RUNTIME_STATE_STATE_UNIT_FRAGMENT_METADATA_SIZE
        + (size_t)fragment->payload_size;
    *fragment_payload = source
        + RUNTIME_STATE_STATE_UNIT_FRAGMENT_METADATA_SIZE;
    if (fragment->payload_size > RUNTIME_STATE_MAX_FRAGMENT_BYTES
        || decoded_size > source_size
        || !runtime_state_valid_state_unit_fragment(
            fragment, *fragment_payload)) {
        return false;
    }
    *record_size = decoded_size;
    return true;
}

static inline bool runtime_state_valid_state_unit_fragment_records(
    const uint8_t *records,
    uint32_t record_count,
    uint32_t record_bytes
) {
    size_t offset = 0U;
    uint32_t index;
    if (records == NULL || record_count == 0U || record_bytes == 0U
        || record_bytes > RUNTIME_STATE_STATE_UNIT_FRAGMENT_METADATA_SIZE
                + RUNTIME_STATE_MAX_FRAGMENT_BYTES) {
        return false;
    }
    for (index = 0U; index < record_count; ++index) {
        struct runtime_state_state_unit_fragment fragment;
        const uint8_t *fragment_payload = NULL;
        size_t record_size = 0U;
        if (!runtime_state_decode_state_unit_fragment_record(
                records + offset, (size_t)record_bytes - offset,
                &fragment, &fragment_payload, &record_size)) {
            return false;
        }
        offset += record_size;
    }
    return offset == (size_t)record_bytes;
}

static inline bool runtime_state_encode_state_unit_fragment_batch_frame(
    uint8_t *destination,
    size_t destination_size,
    const uint8_t *records,
    uint32_t record_count,
    uint32_t record_bytes,
    uint64_t sequence,
    size_t *frame_size
) {
    const size_t payload_size
        = RUNTIME_STATE_STATE_UNIT_FRAGMENT_BATCH_PREFIX_SIZE
            + (size_t)record_bytes;
    uint8_t *payload;
    if (frame_size == NULL
        || !runtime_state_valid_state_unit_fragment_records(
            records, record_count, record_bytes)
        || destination_size < RUNTIME_STATE_HEADER_SIZE + payload_size
        || !runtime_state_encode_header_for_channel(
            destination, destination_size,
            RUNTIME_STATE_FRAME_STATE_UNIT_FRAGMENT_BATCH,
            (uint32_t)payload_size, RUNTIME_STATE_CHANNEL_STATE_UNIT,
            sequence)) {
        return false;
    }
    payload = destination + RUNTIME_STATE_HEADER_SIZE;
    runtime_state_store_le32(payload + 0U, record_count);
    runtime_state_store_le32(payload + 4U, record_bytes);
    memcpy(
        payload + RUNTIME_STATE_STATE_UNIT_FRAGMENT_BATCH_PREFIX_SIZE,
        records,
        record_bytes);
    *frame_size = RUNTIME_STATE_HEADER_SIZE + payload_size;
    return true;
}

static inline bool runtime_state_encode_state_unit_seal_frame(
    uint8_t *destination,
    size_t destination_size,
    const struct runtime_state_state_unit_seal *seal,
    uint64_t sequence
) {
    uint8_t *payload;
    if (destination_size < RUNTIME_STATE_STATE_UNIT_SEAL_FRAME_SIZE
        || !runtime_state_valid_state_unit_seal(seal)
        || !runtime_state_encode_header_for_channel(
            destination, destination_size,
            RUNTIME_STATE_FRAME_STATE_UNIT_SEAL,
            (uint32_t)RUNTIME_STATE_STATE_UNIT_SEAL_SIZE,
            RUNTIME_STATE_CHANNEL_STATE_UNIT, sequence)) {
        return false;
    }
    payload = destination + RUNTIME_STATE_HEADER_SIZE;
    runtime_state_store_le64(payload + 0U, seal->apply_call_id);
    runtime_state_store_le32(payload + 8U, seal->expected_fragment_count);
    runtime_state_store_le32(payload + 12U, seal->expected_token_cell_count);
    runtime_state_store_le32(payload + 16U, seal->stream_mask);
    runtime_state_store_le32(payload + 20U, seal->final_writer_state);
    runtime_state_store_le64(payload + 24U, seal->layout_generation);
    runtime_state_store_le64(payload + 32U, seal->backing_generation);
    runtime_state_store_le64(payload + 40U, seal->mutation_generation);
    runtime_state_store_le64(payload + 48U, seal->copy_byte_count);
    runtime_state_store_le64(payload + 56U, seal->reserve_failure_baseline);
    runtime_state_store_le64(payload + 64U, seal->reserve_failure_final);
    runtime_state_store_le64(payload + 72U, seal->writer_return_monotonic_ns);
    return true;
}

static inline bool runtime_state_encode_topology_frame(
    uint8_t *destination, size_t destination_size,
    const struct runtime_state_topology_event *event, uint64_t sequence
) {
    uint8_t *payload;
    if (destination_size < RUNTIME_STATE_TOPOLOGY_FRAME_SIZE || !runtime_state_valid_topology(event)
        || !runtime_state_encode_header(destination, destination_size, RUNTIME_STATE_FRAME_TOPOLOGY, sequence)) {
        return false;
    }
    payload = destination + RUNTIME_STATE_HEADER_SIZE;
    runtime_state_store_le64(payload + 0U, event->event_id);
    runtime_state_store_le32(payload + 8U, event->tgid);
    runtime_state_store_le32(payload + 12U, event->tid);
    runtime_state_store_le64(payload + 16U, event->monotonic_ns);
    runtime_state_store_le32(payload + 24U, event->n_ctx);
    runtime_state_store_le32(payload + 28U, event->n_stream);
    runtime_state_store_le32(payload + 32U, event->n_seq_max);
    runtime_state_store_le32(payload + 36U, event->n_layer);
    runtime_state_store_le32(payload + 40U, event->v_trans);
    runtime_state_store_le32(payload + 44U, event->n_pos_per_embd);
    return true;
}

static inline bool runtime_state_encode_tensor_frame(
    uint8_t *destination, size_t destination_size,
    const struct runtime_state_tensor_layout_event *event, uint64_t sequence
) {
    uint8_t *payload;
    if (destination_size < RUNTIME_STATE_TENSOR_FRAME_SIZE || !runtime_state_valid_tensor(event)
        || !runtime_state_encode_header(destination, destination_size, RUNTIME_STATE_FRAME_TENSOR_LAYOUT, sequence)) {
        return false;
    }
    payload = destination + RUNTIME_STATE_HEADER_SIZE;
    runtime_state_store_le64(payload + 0U, event->event_id);
    runtime_state_store_le32(payload + 8U, event->tgid);
    runtime_state_store_le32(payload + 12U, event->tid);
    runtime_state_store_le64(payload + 16U, event->monotonic_ns);
    runtime_state_store_le32(payload + 24U, event->kind);
    runtime_state_store_le32(payload + 28U, event->layer);
    runtime_state_store_le32(payload + 32U, event->stream_index);
    runtime_state_store_le32(payload + 36U, event->ggml_type_id);
    runtime_state_store_le64(payload + 40U, event->tensor_address);
    runtime_state_store_le64(payload + 48U, event->row_size);
    runtime_state_store_le64(payload + 56U, event->cell_capacity);
    runtime_state_store_le64(payload + 64U, event->element_size);
    runtime_state_store_le64(payload + 72U, event->embedding_width);
    return true;
}

static inline bool runtime_state_encode_backing_frame(
    uint8_t *destination, size_t destination_size,
    const struct runtime_state_kv_backing_event *event, uint64_t sequence
) {
    uint8_t *payload;
    if (destination_size < RUNTIME_STATE_BACKING_FRAME_SIZE || !runtime_state_valid_backing(event)
        || !runtime_state_encode_header(destination, destination_size, RUNTIME_STATE_FRAME_KV_BACKING, sequence)) {
        return false;
    }
    payload = destination + RUNTIME_STATE_HEADER_SIZE;
    runtime_state_store_le64(payload + 0U, event->event_id);
    runtime_state_store_le32(payload + 8U, event->tgid);
    runtime_state_store_le32(payload + 12U, event->tid);
    runtime_state_store_le64(payload + 16U, event->monotonic_ns);
    runtime_state_store_le64(payload + 24U, event->base_address);
    runtime_state_store_le64(payload + 32U, event->size_bytes);
    return true;
}

static inline bool runtime_state_encode_token_frame(
    uint8_t *destination, size_t destination_size,
    const struct runtime_state_token_cell_event *event, uint64_t sequence
) {
    uint8_t *payload;
    if (destination_size < RUNTIME_STATE_TOKEN_FRAME_SIZE || !runtime_state_valid_token(event)
        || !runtime_state_encode_header(destination, destination_size, RUNTIME_STATE_FRAME_TOKEN_CELL, sequence)) {
        return false;
    }
    payload = destination + RUNTIME_STATE_HEADER_SIZE;
    runtime_state_store_le64(payload + 0U, event->apply_call_id);
    runtime_state_store_le32(payload + 8U, event->tgid);
    runtime_state_store_le32(payload + 12U, event->tid);
    runtime_state_store_le64(payload + 16U, event->monotonic_ns);
    runtime_state_store_le32(payload + 24U, event->seq_id);
    runtime_state_store_le32(payload + 28U, event->stream_index);
    runtime_state_store_le64(payload + 32U, event->token_position);
    runtime_state_store_le64(payload + 40U, event->cell_index);
    return true;
}

static inline bool runtime_state_encode_token_range_frame(
    uint8_t *destination, size_t destination_size,
    const struct runtime_state_token_cell_range_event *event,
    uint64_t sequence
) {
    uint8_t *payload;
    if (destination_size < RUNTIME_STATE_TOKEN_CELL_RANGE_FRAME_SIZE
        || !runtime_state_valid_token_range(event)
        || !runtime_state_encode_header(
            destination, destination_size,
            RUNTIME_STATE_FRAME_TOKEN_CELL_RANGE, sequence)) {
        return false;
    }
    payload = destination + RUNTIME_STATE_HEADER_SIZE;
    runtime_state_store_le64(payload + 0U, event->apply_call_id);
    runtime_state_store_le32(payload + 8U, event->tgid);
    runtime_state_store_le32(payload + 12U, event->tid);
    runtime_state_store_le64(payload + 16U, event->monotonic_ns);
    runtime_state_store_le32(payload + 24U, event->seq_id);
    runtime_state_store_le32(payload + 28U, event->stream_index);
    runtime_state_store_le64(payload + 32U, event->first_token_position);
    runtime_state_store_le64(payload + 40U, event->first_cell_index);
    runtime_state_store_le32(payload + 48U, event->token_count);
    runtime_state_store_le32(payload + 52U, 0U);
    return true;
}

static inline bool runtime_state_encode_apply_complete_frame(
    uint8_t *destination, size_t destination_size,
    const struct runtime_state_apply_complete_event *event, uint64_t sequence
) {
    uint8_t *payload;
    if (destination_size < RUNTIME_STATE_APPLY_COMPLETE_FRAME_SIZE
        || !runtime_state_valid_apply_complete(event)
        || !runtime_state_encode_header(
            destination, destination_size,
            RUNTIME_STATE_FRAME_APPLY_COMPLETE, sequence)) {
        return false;
    }
    payload = destination + RUNTIME_STATE_HEADER_SIZE;
    runtime_state_store_le64(payload + 0U, event->apply_call_id);
    runtime_state_store_le32(payload + 8U, event->tgid);
    runtime_state_store_le32(payload + 12U, event->tid);
    runtime_state_store_le64(payload + 16U, event->monotonic_ns);
    return true;
}

static inline bool runtime_state_encode_mutation_frame(
    uint8_t *destination, size_t destination_size,
    const struct runtime_state_mutation_event *event, uint64_t sequence
) {
    uint8_t *payload;
    if (destination_size < RUNTIME_STATE_MUTATION_FRAME_SIZE
        || !runtime_state_valid_mutation(event)
        || !runtime_state_encode_header(
            destination, destination_size,
            RUNTIME_STATE_FRAME_MUTATION, sequence)) {
        return false;
    }
    payload = destination + RUNTIME_STATE_HEADER_SIZE;
    runtime_state_store_le64(payload + 0U, event->event_id);
    runtime_state_store_le32(payload + 8U, event->tgid);
    runtime_state_store_le32(payload + 12U, event->tid);
    runtime_state_store_le64(payload + 16U, event->monotonic_ns);
    runtime_state_store_le32(payload + 24U, event->mutation_kind);
    runtime_state_store_le32(payload + 28U, 0U);
    return true;
}

static inline bool runtime_state_encode_stream_version_frame(
    uint8_t *destination, size_t destination_size,
    const struct runtime_state_stream_version_snapshot *snapshot,
    uint64_t sequence
) {
    uint8_t *payload;
    size_t index;
    if (destination_size < RUNTIME_STATE_STREAM_VERSION_FRAME_SIZE
        || !runtime_state_valid_stream_version(snapshot)
        || !runtime_state_encode_header(
            destination, destination_size,
            RUNTIME_STATE_FRAME_STREAM_VERSION, sequence)) {
        return false;
    }
    payload = destination + RUNTIME_STATE_HEADER_SIZE;
    runtime_state_store_le64(payload + 0U, snapshot->snapshot_id);
    runtime_state_store_le64(payload + 8U, snapshot->mutation_generation);
    runtime_state_store_le64(payload + 16U, snapshot->backing_generation);
    runtime_state_store_le64(payload + 24U, snapshot->layout_generation);
    runtime_state_store_le32(payload + 32U, snapshot->active_stream_count);
    runtime_state_store_le32(payload + 36U, 0U);
    for (index = 0U; index < RUNTIME_STATE_VERSION_STREAM_COUNT; ++index) {
        runtime_state_store_le64(
            payload + 40U + index * 8U,
            snapshot->stream_apply_call_ids[index]);
        runtime_state_store_le32(
            payload + 104U + index * 4U,
            snapshot->stream_states[index]);
    }
    return true;
}

static inline bool runtime_state_encode_counters_frame(
    uint8_t *destination, size_t destination_size,
    const struct runtime_state_terminal_counters *terminal,
    uint16_t frame_type,
    uint64_t sequence
) {
    uint64_t values[17];
    uint8_t *payload;
    size_t index;
    if (terminal == NULL || destination_size < RUNTIME_STATE_COUNTERS_FRAME_SIZE
        || (frame_type != RUNTIME_STATE_FRAME_WATERMARK
            && frame_type != RUNTIME_STATE_FRAME_TERMINAL)
        || !runtime_state_encode_header(
            destination, destination_size, frame_type, sequence)) {
        return false;
    }
    memcpy(values, terminal, sizeof(values));
    payload = destination + RUNTIME_STATE_HEADER_SIZE;
    for (index = 0U; index < 17U; ++index) {
        runtime_state_store_le64(payload + index * 8U, values[index]);
    }
    return true;
}

static inline bool runtime_state_encode_watermark_frame(
    uint8_t *destination, size_t destination_size,
    const struct runtime_state_terminal_counters *counters, uint64_t sequence
) {
    return runtime_state_encode_counters_frame(
        destination, destination_size, counters,
        RUNTIME_STATE_FRAME_WATERMARK, sequence);
}

static inline bool runtime_state_encode_terminal_frame(
    uint8_t *destination, size_t destination_size,
    const struct runtime_state_terminal_counters *terminal, uint64_t sequence
) {
    return runtime_state_encode_counters_frame(
        destination, destination_size, terminal,
        RUNTIME_STATE_FRAME_TERMINAL, sequence);
}

static inline bool runtime_state_decode_frame_header(
    const uint8_t *source,
    size_t source_size,
    uint16_t expected_type,
    struct runtime_state_frame_header *header
) {
    const size_t expected_size = RUNTIME_STATE_HEADER_SIZE
        + runtime_state_payload_size(expected_type);
    return source != NULL && header != NULL
        && source_size == expected_size
        && runtime_state_decode_header(source, source_size, header)
        && header->frame_type == expected_type;
}

static inline bool runtime_state_decode_topology_frame(
    const uint8_t *source,
    size_t source_size,
    struct runtime_state_topology_event *event
) {
    struct runtime_state_frame_header header;
    const uint8_t *payload;
    if (event == NULL || !runtime_state_decode_frame_header(
            source, source_size, RUNTIME_STATE_FRAME_TOPOLOGY, &header)) {
        return false;
    }
    payload = source + RUNTIME_STATE_HEADER_SIZE;
    event->event_id = runtime_state_load_le64(payload + 0U);
    event->tgid = runtime_state_load_le32(payload + 8U);
    event->tid = runtime_state_load_le32(payload + 12U);
    event->monotonic_ns = runtime_state_load_le64(payload + 16U);
    event->n_ctx = runtime_state_load_le32(payload + 24U);
    event->n_stream = runtime_state_load_le32(payload + 28U);
    event->n_seq_max = runtime_state_load_le32(payload + 32U);
    event->n_layer = runtime_state_load_le32(payload + 36U);
    event->v_trans = runtime_state_load_le32(payload + 40U);
    event->n_pos_per_embd = runtime_state_load_le32(payload + 44U);
    return runtime_state_valid_topology(event);
}

static inline bool runtime_state_decode_tensor_frame(
    const uint8_t *source,
    size_t source_size,
    struct runtime_state_tensor_layout_event *event
) {
    struct runtime_state_frame_header header;
    const uint8_t *payload;
    if (event == NULL || !runtime_state_decode_frame_header(
            source, source_size, RUNTIME_STATE_FRAME_TENSOR_LAYOUT, &header)) {
        return false;
    }
    payload = source + RUNTIME_STATE_HEADER_SIZE;
    event->event_id = runtime_state_load_le64(payload + 0U);
    event->tgid = runtime_state_load_le32(payload + 8U);
    event->tid = runtime_state_load_le32(payload + 12U);
    event->monotonic_ns = runtime_state_load_le64(payload + 16U);
    event->kind = runtime_state_load_le32(payload + 24U);
    event->layer = runtime_state_load_le32(payload + 28U);
    event->stream_index = runtime_state_load_le32(payload + 32U);
    event->ggml_type_id = runtime_state_load_le32(payload + 36U);
    event->tensor_address = runtime_state_load_le64(payload + 40U);
    event->row_size = runtime_state_load_le64(payload + 48U);
    event->cell_capacity = runtime_state_load_le64(payload + 56U);
    event->element_size = runtime_state_load_le64(payload + 64U);
    event->embedding_width = runtime_state_load_le64(payload + 72U);
    return runtime_state_valid_tensor(event);
}

static inline bool runtime_state_decode_backing_frame(
    const uint8_t *source,
    size_t source_size,
    struct runtime_state_kv_backing_event *event
) {
    struct runtime_state_frame_header header;
    const uint8_t *payload;
    if (event == NULL || !runtime_state_decode_frame_header(
            source, source_size, RUNTIME_STATE_FRAME_KV_BACKING, &header)) {
        return false;
    }
    payload = source + RUNTIME_STATE_HEADER_SIZE;
    event->event_id = runtime_state_load_le64(payload + 0U);
    event->tgid = runtime_state_load_le32(payload + 8U);
    event->tid = runtime_state_load_le32(payload + 12U);
    event->monotonic_ns = runtime_state_load_le64(payload + 16U);
    event->base_address = runtime_state_load_le64(payload + 24U);
    event->size_bytes = runtime_state_load_le64(payload + 32U);
    return runtime_state_valid_backing(event);
}

static inline bool runtime_state_decode_token_frame(
    const uint8_t *source,
    size_t source_size,
    struct runtime_state_token_cell_event *event
) {
    struct runtime_state_frame_header header;
    const uint8_t *payload;
    if (event == NULL || !runtime_state_decode_frame_header(
            source, source_size, RUNTIME_STATE_FRAME_TOKEN_CELL, &header)) {
        return false;
    }
    payload = source + RUNTIME_STATE_HEADER_SIZE;
    event->apply_call_id = runtime_state_load_le64(payload + 0U);
    event->tgid = runtime_state_load_le32(payload + 8U);
    event->tid = runtime_state_load_le32(payload + 12U);
    event->monotonic_ns = runtime_state_load_le64(payload + 16U);
    event->seq_id = runtime_state_load_le32(payload + 24U);
    event->stream_index = runtime_state_load_le32(payload + 28U);
    event->token_position = runtime_state_load_le64(payload + 32U);
    event->cell_index = runtime_state_load_le64(payload + 40U);
    return runtime_state_valid_token(event);
}

static inline bool runtime_state_decode_token_range_frame(
    const uint8_t *source,
    size_t source_size,
    struct runtime_state_token_cell_range_event *event
) {
    struct runtime_state_frame_header header;
    const uint8_t *payload;
    if (event == NULL || !runtime_state_decode_frame_header(
            source, source_size,
            RUNTIME_STATE_FRAME_TOKEN_CELL_RANGE, &header)) {
        return false;
    }
    payload = source + RUNTIME_STATE_HEADER_SIZE;
    event->apply_call_id = runtime_state_load_le64(payload + 0U);
    event->tgid = runtime_state_load_le32(payload + 8U);
    event->tid = runtime_state_load_le32(payload + 12U);
    event->monotonic_ns = runtime_state_load_le64(payload + 16U);
    event->seq_id = runtime_state_load_le32(payload + 24U);
    event->stream_index = runtime_state_load_le32(payload + 28U);
    event->first_token_position = runtime_state_load_le64(payload + 32U);
    event->first_cell_index = runtime_state_load_le64(payload + 40U);
    event->token_count = runtime_state_load_le32(payload + 48U);
    event->reserved = runtime_state_load_le32(payload + 52U);
    return runtime_state_valid_token_range(event);
}

static inline bool runtime_state_decode_apply_complete_frame(
    const uint8_t *source,
    size_t source_size,
    struct runtime_state_apply_complete_event *event
) {
    struct runtime_state_frame_header header;
    const uint8_t *payload;
    if (event == NULL || !runtime_state_decode_frame_header(
            source, source_size, RUNTIME_STATE_FRAME_APPLY_COMPLETE, &header)) {
        return false;
    }
    payload = source + RUNTIME_STATE_HEADER_SIZE;
    event->apply_call_id = runtime_state_load_le64(payload + 0U);
    event->tgid = runtime_state_load_le32(payload + 8U);
    event->tid = runtime_state_load_le32(payload + 12U);
    event->monotonic_ns = runtime_state_load_le64(payload + 16U);
    return runtime_state_valid_apply_complete(event);
}

static inline bool runtime_state_decode_mutation_frame(
    const uint8_t *source,
    size_t source_size,
    struct runtime_state_mutation_event *event
) {
    struct runtime_state_frame_header header;
    const uint8_t *payload;
    if (event == NULL || !runtime_state_decode_frame_header(
            source, source_size, RUNTIME_STATE_FRAME_MUTATION, &header)) {
        return false;
    }
    payload = source + RUNTIME_STATE_HEADER_SIZE;
    event->event_id = runtime_state_load_le64(payload + 0U);
    event->tgid = runtime_state_load_le32(payload + 8U);
    event->tid = runtime_state_load_le32(payload + 12U);
    event->monotonic_ns = runtime_state_load_le64(payload + 16U);
    event->mutation_kind = runtime_state_load_le32(payload + 24U);
    event->reserved = runtime_state_load_le32(payload + 28U);
    return runtime_state_valid_mutation(event);
}

static inline bool runtime_state_decode_stream_version_frame(
    const uint8_t *source,
    size_t source_size,
    struct runtime_state_stream_version_snapshot *snapshot
) {
    struct runtime_state_frame_header header;
    const uint8_t *payload;
    size_t index;
    if (snapshot == NULL || !runtime_state_decode_frame_header(
            source, source_size, RUNTIME_STATE_FRAME_STREAM_VERSION, &header)) {
        return false;
    }
    payload = source + RUNTIME_STATE_HEADER_SIZE;
    snapshot->snapshot_id = runtime_state_load_le64(payload + 0U);
    snapshot->mutation_generation = runtime_state_load_le64(payload + 8U);
    snapshot->backing_generation = runtime_state_load_le64(payload + 16U);
    snapshot->layout_generation = runtime_state_load_le64(payload + 24U);
    snapshot->active_stream_count = runtime_state_load_le32(payload + 32U);
    snapshot->reserved = runtime_state_load_le32(payload + 36U);
    for (index = 0U; index < RUNTIME_STATE_VERSION_STREAM_COUNT; ++index) {
        snapshot->stream_apply_call_ids[index] = runtime_state_load_le64(
            payload + 40U + index * 8U);
        snapshot->stream_states[index] = runtime_state_load_le32(
            payload + 104U + index * 4U);
    }
    return runtime_state_valid_stream_version(snapshot);
}

static inline bool runtime_state_decode_state_unit_fragment_frame(
    const uint8_t *source,
    size_t source_size,
    struct runtime_state_state_unit_fragment *fragment,
    const uint8_t **fragment_payload
) {
    struct runtime_state_frame_header header;
    const uint8_t *payload;
    size_t record_size = 0U;
    if (fragment == NULL || fragment_payload == NULL
        || !runtime_state_decode_header(source, source_size, &header)
        || header.frame_type != RUNTIME_STATE_FRAME_STATE_UNIT_FRAGMENT
        || source_size != RUNTIME_STATE_HEADER_SIZE + header.payload_length) {
        return false;
    }
    payload = source + RUNTIME_STATE_HEADER_SIZE;
    return runtime_state_decode_state_unit_fragment_record(
            payload, header.payload_length, fragment,
            fragment_payload, &record_size)
        && record_size == header.payload_length;
}

static inline bool runtime_state_decode_state_unit_fragment_batch_frame(
    const uint8_t *source,
    size_t source_size,
    const uint8_t **records,
    uint32_t *record_count,
    uint32_t *record_bytes
) {
    struct runtime_state_frame_header header;
    const uint8_t *payload;
    if (records == NULL || record_count == NULL || record_bytes == NULL
        || !runtime_state_decode_header(source, source_size, &header)
        || header.frame_type
            != RUNTIME_STATE_FRAME_STATE_UNIT_FRAGMENT_BATCH
        || source_size != RUNTIME_STATE_HEADER_SIZE + header.payload_length) {
        return false;
    }
    payload = source + RUNTIME_STATE_HEADER_SIZE;
    *record_count = runtime_state_load_le32(payload + 0U);
    *record_bytes = runtime_state_load_le32(payload + 4U);
    *records = payload
        + RUNTIME_STATE_STATE_UNIT_FRAGMENT_BATCH_PREFIX_SIZE;
    return header.payload_length
            == RUNTIME_STATE_STATE_UNIT_FRAGMENT_BATCH_PREFIX_SIZE
                + (size_t)*record_bytes
        && runtime_state_valid_state_unit_fragment_records(
            *records, *record_count, *record_bytes);
}

static inline bool runtime_state_decode_state_unit_schedule_batch_frame(
    const uint8_t *source,
    size_t source_size,
    struct runtime_state_state_unit_schedule_batch *batch,
    const uint8_t **packed_payload
) {
    struct runtime_state_frame_header header;
    const uint8_t *payload;
    uint64_t expected_payload_size = 0U;
    uint64_t row_size;
    uint64_t fragments_per_tensor;
    uint32_t record_index;
    uint32_t units_per_fragment;
    if (batch == NULL || packed_payload == NULL
        || !runtime_state_decode_header(source, source_size, &header)
        || header.frame_type
            != RUNTIME_STATE_FRAME_STATE_UNIT_SCHEDULE_BATCH
        || source_size != RUNTIME_STATE_HEADER_SIZE + header.payload_length) {
        return false;
    }
    payload = source + RUNTIME_STATE_HEADER_SIZE;
    batch->record_count = runtime_state_load_le32(payload + 0U);
    batch->payload_size = runtime_state_load_le32(payload + 4U);
    batch->apply_call_id = runtime_state_load_le64(payload + 8U);
    batch->first_fragment_index = runtime_state_load_le32(payload + 16U);
    batch->first_flat_index = runtime_state_load_le32(payload + 20U);
    batch->seq_id = runtime_state_load_le32(payload + 24U);
    batch->stream_index = runtime_state_load_le32(payload + 28U);
    batch->tensor_kind = runtime_state_load_le32(payload + 32U);
    batch->fragments_per_tensor = runtime_state_load_le32(payload + 36U);
    batch->first_token_position = runtime_state_load_le64(payload + 40U);
    batch->first_cell_index = runtime_state_load_le64(payload + 48U);
    batch->chunks_per_dimension = runtime_state_load_le32(payload + 56U);
    batch->token_count = runtime_state_load_le32(payload + 60U);
    batch->row_size = runtime_state_load_le32(payload + 64U);
    batch->cell_capacity = runtime_state_load_le32(payload + 68U);
    batch->element_size = runtime_state_load_le32(payload + 72U);
    batch->embedding_width = runtime_state_load_le32(payload + 76U);
    batch->transposed_value = runtime_state_load_le32(payload + 80U);
    batch->reserved = runtime_state_load_le32(payload + 84U);
    batch->writer_monotonic_ns = runtime_state_load_le64(payload + 88U);
    *packed_payload = payload
        + RUNTIME_STATE_STATE_UNIT_SCHEDULE_BATCH_PREFIX_SIZE;
    if (batch->record_count == 0U || batch->record_count > 32U
        || batch->payload_size == 0U
        || batch->payload_size > RUNTIME_STATE_MAX_FRAGMENT_BYTES
        || header.payload_length
            != RUNTIME_STATE_STATE_UNIT_SCHEDULE_BATCH_PREFIX_SIZE
                + batch->payload_size
        || batch->apply_call_id == 0U || batch->stream_index >= 8U
        || batch->tensor_kind != RUNTIME_STATE_TENSOR_KIND_V
        || batch->fragments_per_tensor == 0U
        || batch->chunks_per_dimension == 0U
        || batch->token_count == 0U || batch->row_size == 0U
        || batch->cell_capacity == 0U || batch->element_size == 0U
        || batch->embedding_width == 0U
        || batch->transposed_value != 1U || batch->reserved != 0U
        || batch->first_cell_index >= batch->cell_capacity
        || batch->token_count
            > batch->cell_capacity - batch->first_cell_index
        || batch->first_fragment_index
            > UINT32_MAX - batch->record_count
        || batch->first_flat_index > UINT32_MAX - batch->record_count
        || batch->first_token_position
            > UINT64_MAX - batch->token_count) {
        return false;
    }
    row_size = (uint64_t)batch->element_size * batch->embedding_width;
    fragments_per_tensor = (uint64_t)batch->chunks_per_dimension
        * batch->embedding_width;
    if (row_size != batch->row_size
        || fragments_per_tensor != batch->fragments_per_tensor) {
        return false;
    }
    units_per_fragment = RUNTIME_STATE_MAX_FRAGMENT_BYTES
        / batch->element_size;
    if (units_per_fragment == 0U) {
        return false;
    }
    for (record_index = 0U;
            record_index < batch->record_count;
            ++record_index) {
        const uint32_t flat_index = batch->first_flat_index + record_index;
        const uint32_t fragment_in_tensor
            = flat_index % batch->fragments_per_tensor;
        const uint32_t chunk_index
            = fragment_in_tensor % batch->chunks_per_dimension;
        uint32_t cell_offset;
        uint32_t cell_count;
        uint32_t record_payload_size;
        if (chunk_index > UINT32_MAX / units_per_fragment) {
            return false;
        }
        cell_offset = chunk_index * units_per_fragment;
        if (cell_offset >= batch->token_count) {
            return false;
        }
        cell_count = batch->token_count - cell_offset;
        if (cell_count > units_per_fragment) {
            cell_count = units_per_fragment;
        }
        if (cell_count > UINT32_MAX / batch->element_size) {
            return false;
        }
        record_payload_size = cell_count * batch->element_size;
        if (record_payload_size == 0U || record_payload_size > 128U) {
            return false;
        }
        expected_payload_size += record_payload_size;
    }
    return expected_payload_size == batch->payload_size;
}

static inline bool runtime_state_decode_state_unit_seal_frame(
    const uint8_t *source,
    size_t source_size,
    struct runtime_state_state_unit_seal *seal
) {
    struct runtime_state_frame_header header;
    const uint8_t *payload;
    if (seal == NULL || !runtime_state_decode_frame_header(
            source, source_size,
            RUNTIME_STATE_FRAME_STATE_UNIT_SEAL, &header)) {
        return false;
    }
    payload = source + RUNTIME_STATE_HEADER_SIZE;
    seal->apply_call_id = runtime_state_load_le64(payload + 0U);
    seal->expected_fragment_count = runtime_state_load_le32(payload + 8U);
    seal->expected_token_cell_count = runtime_state_load_le32(payload + 12U);
    seal->stream_mask = runtime_state_load_le32(payload + 16U);
    seal->final_writer_state = runtime_state_load_le32(payload + 20U);
    seal->layout_generation = runtime_state_load_le64(payload + 24U);
    seal->backing_generation = runtime_state_load_le64(payload + 32U);
    seal->mutation_generation = runtime_state_load_le64(payload + 40U);
    seal->copy_byte_count = runtime_state_load_le64(payload + 48U);
    seal->reserve_failure_baseline = runtime_state_load_le64(payload + 56U);
    seal->reserve_failure_final = runtime_state_load_le64(payload + 64U);
    seal->writer_return_monotonic_ns = runtime_state_load_le64(payload + 72U);
    return runtime_state_valid_state_unit_seal(seal);
}

static inline bool runtime_state_decode_counters_frame(
    const uint8_t *source,
    size_t source_size,
    uint16_t expected_type,
    struct runtime_state_terminal_counters *counters
) {
    struct runtime_state_frame_header header;
    uint64_t values[17];
    const uint8_t *payload;
    size_t index;
    if (counters == NULL
        || (expected_type != RUNTIME_STATE_FRAME_WATERMARK
            && expected_type != RUNTIME_STATE_FRAME_TERMINAL)
        || !runtime_state_decode_frame_header(
            source, source_size, expected_type, &header)) {
        return false;
    }
    payload = source + RUNTIME_STATE_HEADER_SIZE;
    for (index = 0U; index < 17U; ++index) {
        values[index] = runtime_state_load_le64(payload + index * 8U);
    }
    memcpy(counters, values, sizeof(values));
    return true;
}

static inline bool runtime_state_decode_watermark_frame(
    const uint8_t *source,
    size_t source_size,
    struct runtime_state_terminal_counters *counters
) {
    return runtime_state_decode_counters_frame(
        source, source_size, RUNTIME_STATE_FRAME_WATERMARK, counters);
}

static inline bool runtime_state_decode_terminal_frame(
    const uint8_t *source,
    size_t source_size,
    struct runtime_state_terminal_counters *terminal
) {
    return runtime_state_decode_counters_frame(
        source, source_size, RUNTIME_STATE_FRAME_TERMINAL, terminal);
}

static inline bool runtime_state_validate_stream(const uint8_t *stream, size_t stream_size) {
    size_t offset = 0U;
    uint64_t sequences[RUNTIME_STATE_CHANNEL_COUNT] = {1U, 1U};
    bool terminal_seen = false;
    if (stream == NULL || stream_size == 0U) {
        return false;
    }
    while (offset < stream_size) {
        struct runtime_state_frame_header header;
        size_t frame_size;
        if (!runtime_state_decode_header(stream + offset, stream_size - offset, &header)
            || header.channel_id >= RUNTIME_STATE_CHANNEL_COUNT
            || header.frame_sequence != sequences[header.channel_id]
            || terminal_seen) {
            return false;
        }
        frame_size = RUNTIME_STATE_HEADER_SIZE + (size_t)header.payload_length;
        if (frame_size > stream_size - offset) {
            return false;
        }
        switch (header.frame_type) {
            case RUNTIME_STATE_FRAME_TOPOLOGY: {
                struct runtime_state_topology_event event;
                if (!runtime_state_decode_topology_frame(stream + offset, frame_size, &event)) return false;
                break;
            }
            case RUNTIME_STATE_FRAME_TENSOR_LAYOUT: {
                struct runtime_state_tensor_layout_event event;
                if (!runtime_state_decode_tensor_frame(stream + offset, frame_size, &event)) return false;
                break;
            }
            case RUNTIME_STATE_FRAME_KV_BACKING: {
                struct runtime_state_kv_backing_event event;
                if (!runtime_state_decode_backing_frame(stream + offset, frame_size, &event)) return false;
                break;
            }
            case RUNTIME_STATE_FRAME_TOKEN_CELL: {
                struct runtime_state_token_cell_event event;
                if (!runtime_state_decode_token_frame(stream + offset, frame_size, &event)) return false;
                break;
            }
            case RUNTIME_STATE_FRAME_TOKEN_CELL_RANGE: {
                struct runtime_state_token_cell_range_event event;
                if (!runtime_state_decode_token_range_frame(
                        stream + offset, frame_size, &event)) return false;
                break;
            }
            case RUNTIME_STATE_FRAME_APPLY_COMPLETE: {
                struct runtime_state_apply_complete_event event;
                if (!runtime_state_decode_apply_complete_frame(
                        stream + offset, frame_size, &event)) return false;
                break;
            }
            case RUNTIME_STATE_FRAME_MUTATION: {
                struct runtime_state_mutation_event event;
                if (!runtime_state_decode_mutation_frame(
                        stream + offset, frame_size, &event)) return false;
                break;
            }
            case RUNTIME_STATE_FRAME_STREAM_VERSION: {
                struct runtime_state_stream_version_snapshot event;
                if (!runtime_state_decode_stream_version_frame(
                        stream + offset, frame_size, &event)) return false;
                break;
            }
            case RUNTIME_STATE_FRAME_STATE_UNIT_FRAGMENT: {
                struct runtime_state_state_unit_fragment event;
                const uint8_t *payload = NULL;
                if (!runtime_state_decode_state_unit_fragment_frame(
                        stream + offset, frame_size, &event, &payload)) return false;
                break;
            }
            case RUNTIME_STATE_FRAME_STATE_UNIT_FRAGMENT_BATCH: {
                const uint8_t *records = NULL;
                uint32_t record_count = 0U;
                uint32_t record_bytes = 0U;
                if (!runtime_state_decode_state_unit_fragment_batch_frame(
                        stream + offset, frame_size, &records,
                        &record_count, &record_bytes)) return false;
                break;
            }
            case RUNTIME_STATE_FRAME_STATE_UNIT_SCHEDULE_BATCH: {
                struct runtime_state_state_unit_schedule_batch event;
                const uint8_t *payload = NULL;
                if (!runtime_state_decode_state_unit_schedule_batch_frame(
                        stream + offset, frame_size, &event, &payload)) {
                    return false;
                }
                break;
            }
            case RUNTIME_STATE_FRAME_STATE_UNIT_BULK:
                break;
            case RUNTIME_STATE_FRAME_STATE_UNIT_SEAL: {
                struct runtime_state_state_unit_seal event;
                if (!runtime_state_decode_state_unit_seal_frame(
                        stream + offset, frame_size, &event)) return false;
                break;
            }
            case RUNTIME_STATE_FRAME_WATERMARK: {
                struct runtime_state_terminal_counters event;
                if (!runtime_state_decode_watermark_frame(
                        stream + offset, frame_size, &event)) return false;
                break;
            }
            case RUNTIME_STATE_FRAME_TERMINAL: {
                struct runtime_state_terminal_counters event;
                if (!runtime_state_decode_terminal_frame(stream + offset, frame_size, &event)) return false;
                terminal_seen = true;
                if (offset + frame_size != stream_size) return false;
                break;
            }
            default: return false;
        }
        offset += frame_size;
        sequences[header.channel_id] += 1U;
    }
    return terminal_seen;
}

#endif
