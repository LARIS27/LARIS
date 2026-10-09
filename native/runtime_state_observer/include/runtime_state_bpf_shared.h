#ifndef LARIS_RUNTIME_STATE_BPF_SHARED_H
#define LARIS_RUNTIME_STATE_BPF_SHARED_H

#ifdef _WIN32
#include <stdint.h>
typedef uint8_t __u8;
typedef uint32_t __u32;
typedef uint64_t __u64;
#else
#include <linux/types.h>
#endif

#define RUNTIME_STATE_MAX_APPLY_TOKENS 2048U
#define RUNTIME_STATE_MAX_APPLY_STREAMS 8U
#define RUNTIME_STATE_MAX_TENSOR_EVENTS 1024U
#define RUNTIME_STATE_MAX_FRAGMENT_BYTES 4096U
#define RUNTIME_STATE_MAX_BATCH_FRAGMENTS 32U
#define RUNTIME_STATE_MAX_BATCH_FRAGMENT_BYTES 128U
#define RUNTIME_STATE_KERNEL_STATE_UNIT_FRAGMENT_METADATA_SIZE 72U
#define RUNTIME_STATE_KERNEL_STATE_UNIT_FRAGMENT_EVENT_PREFIX_SIZE 80U
#define RUNTIME_STATE_KERNEL_STATE_UNIT_FRAGMENT_BATCH_EVENT_PREFIX_SIZE 2320U
#define RUNTIME_STATE_STREAM_VERSION_BUSY_FLAG ((__u64)1ULL << 63U)
#define RUNTIME_STATE_CAPTURE_METADATA 0U
#define RUNTIME_STATE_CAPTURE_WRITER_BOUNDARY 1U
#define RUNTIME_STATE_CAPTURE_COMPACT_METADATA 2U
#define RUNTIME_STATE_CAPTURE_GATHER_WRITER_BOUNDARY 3U

/* Qualified against llama.cpp daf6bc9 on System V AMD64. */
#define RUNTIME_STATE_CACHE_HPARAMS_OFFSET 16U
#define RUNTIME_STATE_CACHE_V_TRANS_OFFSET 24U
#define RUNTIME_STATE_CACHE_N_SEQ_MAX_OFFSET 28U
#define RUNTIME_STATE_CACHE_N_STREAM_OFFSET 32U
#define RUNTIME_STATE_CACHE_LAYERS_OFFSET 272U
#define RUNTIME_STATE_KV_LAYER_SIZE 72U
#define RUNTIME_STATE_KV_LAYER_K_STREAM_OFFSET 24U
#define RUNTIME_STATE_KV_LAYER_V_STREAM_OFFSET 48U
#define RUNTIME_STATE_HPARAMS_ROPE_TYPE_OFFSET 20936U
#define RUNTIME_STATE_VECTOR_BEGIN_OFFSET 0U
#define RUNTIME_STATE_VECTOR_END_OFFSET 8U
#define RUNTIME_STATE_TENSOR_NE_OFFSET 16U
#define RUNTIME_STATE_TENSOR_NB_OFFSET 48U
#define RUNTIME_STATE_TENSOR_VIEW_SRC_OFFSET 232U
#define RUNTIME_STATE_TENSOR_VIEW_OFFS_OFFSET 240U

#define RUNTIME_STATE_KERNEL_TOPOLOGY 1U
#define RUNTIME_STATE_KERNEL_TENSOR_LAYOUT 2U
#define RUNTIME_STATE_KERNEL_KV_BACKING 3U
#define RUNTIME_STATE_KERNEL_TOKEN_CELL 4U
#define RUNTIME_STATE_KERNEL_APPLY_COMPLETE 5U
#define RUNTIME_STATE_KERNEL_MUTATION 6U
#define RUNTIME_STATE_KERNEL_TOKEN_CELL_RANGE 7U

#define RUNTIME_STATE_KERNEL_STATE_UNIT_FRAGMENT 1U
#define RUNTIME_STATE_KERNEL_STATE_UNIT_SEAL 2U
#define RUNTIME_STATE_KERNEL_STATE_UNIT_FRAGMENT_BATCH 3U
#define RUNTIME_STATE_KERNEL_STATE_UNIT_SCHEDULE_BATCH 4U
#define RUNTIME_STATE_KERNEL_STATE_UNIT_READ_PLAN 5U
#define RUNTIME_STATE_KERNEL_STATE_UNIT_APPLY_READY 6U
#define RUNTIME_STATE_WRITER_COMPLETE 1U

#define RUNTIME_STATE_MUTATION_CACHE_SHIFT 1U
#define RUNTIME_STATE_MUTATION_CONTEXT_SHIFT 2U
#define RUNTIME_STATE_MUTATION_KV_DEFRAGMENTATION 3U
#define RUNTIME_STATE_MUTATION_MULTIMODAL_POSITION 4U
#define RUNTIME_STATE_MUTATION_SEQUENCE_COPY 5U
#define RUNTIME_STATE_MUTATION_SEQUENCE_REMOVE 6U
#define RUNTIME_STATE_MUTATION_SEQUENCE_STREAM_REMAP 7U
#define RUNTIME_STATE_MUTATION_SPECULATIVE_CONTEXT 8U

#define RUNTIME_STATE_TENSOR_KIND_K 1U
#define RUNTIME_STATE_TENSOR_KIND_V 2U

static __inline __u32 runtime_state_tensor_kind_for_ordinal(
    __u32 ordinal,
    __u32 n_stream
) {
    (void)n_stream;
    return ordinal % 2U == 0U
        ? RUNTIME_STATE_TENSOR_KIND_K
        : RUNTIME_STATE_TENSOR_KIND_V;
}

static __inline __u32 runtime_state_tensor_stream_for_ordinal(
    __u32 ordinal,
    __u32 n_stream
) {
    return (ordinal % (2U * n_stream)) / 2U;
}

static __inline __u32 runtime_state_tensor_layer_for_ordinal(
    __u32 ordinal,
    __u32 n_stream
) {
    return ordinal / (2U * n_stream);
}

static __inline __u32 runtime_state_kernel_fragment_batch_can_append(
    __u32 record_count,
    __u32 payload_bytes,
    __u32 next_payload_bytes
) {
    const __u64 maximum_record_bytes
        = RUNTIME_STATE_KERNEL_STATE_UNIT_FRAGMENT_METADATA_SIZE
            + RUNTIME_STATE_MAX_FRAGMENT_BYTES;
    __u64 current_record_bytes;
    __u64 next_record_bytes;
    if (record_count >= RUNTIME_STATE_MAX_BATCH_FRAGMENTS
        || payload_bytes > RUNTIME_STATE_MAX_FRAGMENT_BYTES
        || next_payload_bytes == 0U
        || next_payload_bytes > RUNTIME_STATE_MAX_BATCH_FRAGMENT_BYTES
        || next_payload_bytes
            > RUNTIME_STATE_MAX_FRAGMENT_BYTES - payload_bytes) {
        return 0U;
    }
    current_record_bytes = (__u64)record_count
        * RUNTIME_STATE_KERNEL_STATE_UNIT_FRAGMENT_METADATA_SIZE
        + payload_bytes;
    next_record_bytes
        = RUNTIME_STATE_KERNEL_STATE_UNIT_FRAGMENT_METADATA_SIZE
            + next_payload_bytes;
    return current_record_bytes <= maximum_record_bytes - next_record_bytes
        ? 1U : 0U;
}

static __inline __u32 runtime_state_kernel_schedule_batch_can_append(
    __u32 record_count,
    __u32 payload_bytes,
    __u32 next_payload_bytes
) {
    return record_count < RUNTIME_STATE_MAX_BATCH_FRAGMENTS
            && payload_bytes <= RUNTIME_STATE_MAX_FRAGMENT_BYTES
            && next_payload_bytes != 0U
            && next_payload_bytes <= RUNTIME_STATE_MAX_BATCH_FRAGMENT_BYTES
            && next_payload_bytes
                <= RUNTIME_STATE_MAX_FRAGMENT_BYTES - payload_bytes
        ? 1U : 0U;
}

struct runtime_state_target_config {
    __u32 target_tgid;
    __u32 expected_n_stream;
    __u32 expected_n_seq_max;
    __u32 expected_n_layer;
    __u32 tensor_data_offset;
    __u32 tensor_type_offset;
    __u32 capture_mode;
    __u32 capture_armed;
};

enum runtime_state_compute_phase {
    RUNTIME_STATE_COMPUTE_ACTIVE = 1,
    RUNTIME_STATE_COMPUTE_APPLY_PLANNED = 2,
    RUNTIME_STATE_COMPUTE_APPLY_RETURNED = 3,
};

struct runtime_state_compute_call {
    __u64 context_address;
    __u64 apply_call_id;
    __u32 phase;
    __u32 stream_mask;
    __u32 token_cell_count;
    __u32 reserved;
    __u64 payload_reserve_failure_baseline;
};

struct runtime_state_kernel_topology_event {
    __u64 event_id;
    __u32 tgid;
    __u32 tid;
    __u64 monotonic_ns;
    __u32 n_ctx;
    __u32 n_stream;
    __u32 n_seq_max;
    __u32 n_layer;
    __u32 v_trans;
    __u32 n_pos_per_embd;
};

struct runtime_state_kernel_tensor_layout_event {
    __u64 event_id;
    __u32 tgid;
    __u32 tid;
    __u64 monotonic_ns;
    __u32 kind;
    __u32 layer;
    __u32 stream_index;
    __u32 ggml_type_id;
    __u64 tensor_address;
    __u64 row_size;
    __u64 cell_capacity;
    __u64 element_size;
    __u64 embedding_width;
};

struct runtime_state_kernel_kv_backing_event {
    __u64 event_id;
    __u32 tgid;
    __u32 tid;
    __u64 monotonic_ns;
    __u64 base_address;
    __u64 size_bytes;
};

struct runtime_state_kernel_token_cell_event {
    __u64 apply_call_id;
    __u32 tgid;
    __u32 tid;
    __u64 monotonic_ns;
    __u32 seq_id;
    __u32 stream_index;
    __u64 token_position;
    __u64 cell_index;
};

struct runtime_state_kernel_token_cell_range_event {
    __u64 apply_call_id;
    __u32 tgid;
    __u32 tid;
    __u64 monotonic_ns;
    __u32 seq_id;
    __u32 stream_index;
    __u64 first_token_position;
    __u64 first_cell_index;
    __u32 token_count;
    __u32 reserved;
};

struct runtime_state_kernel_apply_complete_event {
    __u64 apply_call_id;
    __u32 tgid;
    __u32 tid;
    __u64 monotonic_ns;
};

struct runtime_state_kernel_mutation_event {
    __u64 event_id;
    __u32 tgid;
    __u32 tid;
    __u64 monotonic_ns;
    __u32 mutation_kind;
    __u32 reserved;
};

struct runtime_state_kernel_event {
    __u32 event_type;
    __u32 reserved;
    union {
        struct runtime_state_kernel_topology_event topology;
        struct runtime_state_kernel_tensor_layout_event tensor;
        struct runtime_state_kernel_kv_backing_event backing;
        struct runtime_state_kernel_token_cell_event token;
        struct runtime_state_kernel_token_cell_range_event token_range;
        struct runtime_state_kernel_apply_complete_event apply_complete;
        struct runtime_state_kernel_mutation_event mutation;
    } payload;
};

struct runtime_state_kernel_state_unit_fragment {
    __u64 apply_call_id;
    __u32 fragment_index;
    __u32 seq_id;
    __u32 stream_index;
    __u32 tensor_kind;
    __u64 token_position;
    __u64 cell_index;
    __u32 layer;
    __u32 payload_size;
    __u64 tensor_byte_offset;
    __u64 writer_monotonic_ns;
    __u64 span_cell_count;
    __u8 payload[RUNTIME_STATE_MAX_FRAGMENT_BYTES];
};

struct runtime_state_kernel_state_unit_fragment_metadata {
    __u64 apply_call_id;
    __u32 fragment_index;
    __u32 seq_id;
    __u32 stream_index;
    __u32 tensor_kind;
    __u64 token_position;
    __u64 cell_index;
    __u32 layer;
    __u32 payload_size;
    __u64 tensor_byte_offset;
    __u64 writer_monotonic_ns;
    __u64 span_cell_count;
};

struct runtime_state_kernel_state_unit_seal {
    __u64 apply_call_id;
    __u32 expected_fragment_count;
    __u32 expected_token_cell_count;
    __u32 stream_mask;
    __u32 final_writer_state;
    __u64 layout_generation;
    __u64 backing_generation;
    __u64 mutation_generation;
    __u64 copy_byte_count;
    __u64 reserve_failure_baseline;
    __u64 reserve_failure_final;
    __u64 writer_return_monotonic_ns;
};

struct runtime_state_kernel_state_unit_event_header {
    __u32 event_type;
    __u32 reserved;
};

struct runtime_state_kernel_state_unit_fragment_event {
    __u32 event_type;
    __u32 reserved;
    struct runtime_state_kernel_state_unit_fragment payload;
};

struct runtime_state_kernel_state_unit_fragment_batch_event {
    __u32 event_type;
    __u32 reserved;
    __u32 record_count;
    __u32 payload_size;
    struct runtime_state_kernel_state_unit_fragment_metadata
        fragments[RUNTIME_STATE_MAX_BATCH_FRAGMENTS];
    __u8 payload[RUNTIME_STATE_MAX_FRAGMENT_BYTES];
};

struct runtime_state_kernel_state_unit_schedule_batch_event {
    __u32 event_type;
    __u32 reserved;
    __u64 apply_call_id;
    __u64 first_token_position;
    __u64 first_cell_index;
    __u64 writer_monotonic_ns;
    __u32 record_count;
    __u32 payload_size;
    __u32 first_fragment_index;
    __u32 first_flat_index;
    __u32 seq_id;
    __u32 stream_index;
    __u32 tensor_kind;
    __u32 fragments_per_tensor;
    __u32 chunks_per_dimension;
    __u32 token_count;
    __u32 row_size;
    __u32 cell_capacity;
    __u32 element_size;
    __u32 embedding_width;
    __u32 transposed_value;
    __u32 schedule_reserved;
    __u8 payload[RUNTIME_STATE_MAX_FRAGMENT_BYTES];
};

struct runtime_state_kernel_state_unit_read_plan_event {
    __u32 event_type;
    __u32 reserved;
    __u64 apply_call_id;
    __u64 tensor_address;
    __u64 first_token_position;
    __u64 first_cell_index;
    __u64 writer_monotonic_ns;
    __u64 copy_byte_count;
    __u32 first_fragment_index;
    __u32 fragment_count;
    __u32 seq_id;
    __u32 stream_index;
    __u32 tensor_kind;
    __u32 layer;
    __u32 expected_n_layer;
    __u32 token_count;
    __u32 row_size;
    __u32 cell_capacity;
    __u32 element_size;
    __u32 embedding_width;
    __u32 chunks_per_dimension;
    __u32 fragments_per_tensor;
    __u32 transposed_value;
    __u32 plan_reserved;
};

struct runtime_state_kernel_state_unit_apply_ready_event {
    __u32 event_type;
    __u32 reserved;
    struct runtime_state_kernel_state_unit_seal payload;
};

struct runtime_state_kernel_state_unit_seal_event {
    __u32 event_type;
    __u32 reserved;
    struct runtime_state_kernel_state_unit_seal payload;
};

struct runtime_state_bpf_counters {
    __u64 topology_event_count;
    __u64 tensor_layout_event_count;
    __u64 kv_backing_event_count;
    __u64 token_cell_event_count;
    __u64 apply_complete_event_count;
    __u64 mutation_event_count;
    __u64 kernel_event_submit_count;
    __u64 ringbuf_reserve_fail_count;
    __u64 user_memory_read_failure_count;
    __u64 unmatched_call_count;
    __u64 unsupported_mutation_count;
    __u64 mapping_mutation_count;
    __u64 inflight_call_count;
    __u64 active_emitter_count;
    __u64 next_event_id;
    __u64 next_apply_call_id;
    __u64 state_unit_fragment_submit_count;
    __u64 state_unit_seal_submit_count;
    __u64 payload_ringbuf_reserve_fail_count;
    __u64 payload_user_memory_read_failure_count;
};

struct runtime_state_bpf_diagnostics {
    __u64 candidate_missing_count;
    __u64 shape_invalid_count;
    __u64 tensor_data_read_failure_count;
    __u64 tensor_type_read_failure_count;
    __u64 tensor_data_null_count;
    __u64 apply_state_conflict_count;
    __u64 apply_header_read_failure_count;
    __u64 apply_slot_shape_invalid_count;
    __u64 apply_batch_shape_invalid_count;
    __u64 apply_stream_count_zero_count;
    __u64 apply_stream_count_exceeds_count;
    __u64 apply_index_vector_extent_invalid_count;
    __u64 apply_token_count_zero_count;
    __u64 apply_token_count_exceeds_count;
    __u64 apply_token_count_exceeds_sample;
    __u64 apply_token_stream_remainder_count;
    __u64 apply_token_read_failure_count;
    __u64 apply_token_shape_invalid_count;
    __u64 apply_return_unmatched_count;
    __u64 apply_state_delete_failure_count;
    __u64 process_state_conflict_count;
    __u64 process_return_unmatched_count;
    __u64 process_apply_missing_count;
    __u64 process_graph_compute_failed_count;
};

#ifdef __cplusplus
#define RUNTIME_STATE_STATIC_ASSERT static_assert
#else
#define RUNTIME_STATE_STATIC_ASSERT _Static_assert
#endif

RUNTIME_STATE_STATIC_ASSERT(sizeof(struct runtime_state_kernel_topology_event) == 48U, "topology payload size mismatch");
RUNTIME_STATE_STATIC_ASSERT(sizeof(struct runtime_state_kernel_tensor_layout_event) == 80U, "tensor payload size mismatch");
RUNTIME_STATE_STATIC_ASSERT(sizeof(struct runtime_state_kernel_kv_backing_event) == 40U, "backing payload size mismatch");
RUNTIME_STATE_STATIC_ASSERT(sizeof(struct runtime_state_kernel_token_cell_event) == 48U, "token payload size mismatch");
RUNTIME_STATE_STATIC_ASSERT(sizeof(struct runtime_state_kernel_token_cell_range_event) == 56U, "token range payload size mismatch");
RUNTIME_STATE_STATIC_ASSERT(sizeof(struct runtime_state_kernel_apply_complete_event) == 24U, "apply completion payload size mismatch");
RUNTIME_STATE_STATIC_ASSERT(sizeof(struct runtime_state_kernel_mutation_event) == 32U, "mutation payload size mismatch");
RUNTIME_STATE_STATIC_ASSERT(sizeof(struct runtime_state_kernel_event) == 88U, "kernel event size mismatch");
RUNTIME_STATE_STATIC_ASSERT(sizeof(struct runtime_state_kernel_state_unit_fragment) == 4168U, "state-unit fragment size mismatch");
RUNTIME_STATE_STATIC_ASSERT(sizeof(struct runtime_state_kernel_state_unit_fragment_metadata) == RUNTIME_STATE_KERNEL_STATE_UNIT_FRAGMENT_METADATA_SIZE, "state-unit fragment metadata size mismatch");
RUNTIME_STATE_STATIC_ASSERT(sizeof(struct runtime_state_kernel_state_unit_seal) == 80U, "state-unit seal size mismatch");
RUNTIME_STATE_STATIC_ASSERT(sizeof(struct runtime_state_kernel_state_unit_event_header) == 8U, "state-unit event header size mismatch");
RUNTIME_STATE_STATIC_ASSERT(sizeof(struct runtime_state_kernel_state_unit_fragment_event) == 4176U, "state-unit fragment event size mismatch");
RUNTIME_STATE_STATIC_ASSERT(sizeof(struct runtime_state_kernel_state_unit_fragment_batch_event) == 6416U, "state-unit fragment batch event size mismatch");
RUNTIME_STATE_STATIC_ASSERT(sizeof(struct runtime_state_kernel_state_unit_schedule_batch_event) == 4200U, "state-unit schedule batch event size mismatch");
RUNTIME_STATE_STATIC_ASSERT(sizeof(struct runtime_state_kernel_state_unit_read_plan_event) == 120U, "state-unit read plan event size mismatch");
RUNTIME_STATE_STATIC_ASSERT(sizeof(struct runtime_state_kernel_state_unit_apply_ready_event) == 88U, "state-unit apply ready event size mismatch");
RUNTIME_STATE_STATIC_ASSERT(
    RUNTIME_STATE_MAX_BATCH_FRAGMENTS
            * RUNTIME_STATE_MAX_BATCH_FRAGMENT_BYTES
        == RUNTIME_STATE_MAX_FRAGMENT_BYTES,
    "state-unit fragment batch payload slots must fill the scratch payload");
RUNTIME_STATE_STATIC_ASSERT(
    __builtin_offsetof(
        struct runtime_state_kernel_state_unit_fragment_batch_event,
        payload)
        == RUNTIME_STATE_KERNEL_STATE_UNIT_FRAGMENT_BATCH_EVENT_PREFIX_SIZE,
    "state-unit fragment batch event prefix size mismatch");
RUNTIME_STATE_STATIC_ASSERT(
    __builtin_offsetof(
        struct runtime_state_kernel_state_unit_fragment_event,
        payload.payload)
        == RUNTIME_STATE_KERNEL_STATE_UNIT_FRAGMENT_EVENT_PREFIX_SIZE,
    "state-unit fragment event prefix size mismatch");
RUNTIME_STATE_STATIC_ASSERT(sizeof(struct runtime_state_kernel_state_unit_seal_event) == 88U, "state-unit seal event size mismatch");
RUNTIME_STATE_STATIC_ASSERT(sizeof(struct runtime_state_target_config) == 32U, "target configuration size mismatch");
RUNTIME_STATE_STATIC_ASSERT(sizeof(struct runtime_state_compute_call) == 40U, "compute call size mismatch");
RUNTIME_STATE_STATIC_ASSERT(sizeof(struct runtime_state_bpf_counters) == 160U, "BPF counters size mismatch");
RUNTIME_STATE_STATIC_ASSERT(sizeof(struct runtime_state_bpf_diagnostics) == 192U, "BPF diagnostics size mismatch");

#undef RUNTIME_STATE_STATIC_ASSERT

#endif
