#include "runtime_state_bpf_shared.h"
#include "runtime_state_batch.h"
#include "runtime_state_bulk.h"
#include "runtime_state_wire.h"

#include <stdio.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "runtime-state wire check failed at line %d: %s\n", __LINE__, #condition); \
    return 1; \
} } while (0)

static const uint8_t GOLDEN_BACKING[RUNTIME_STATE_BACKING_FRAME_SIZE] = {
    0x4c,0x53,0x52,0x53, 0x05,0x00, 0x03,0x00, 0x28,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00, 0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x08,0x07,0x06,0x05,0x04,0x03,0x02,0x01, 0x04,0x03,0x02,0x01,
    0x44,0x33,0x22,0x11, 0x02,0x02,0x02,0x02,0x01,0x01,0x01,0x01,
    0x88,0x77,0x66,0x55,0x44,0x33,0x22,0x11, 0x08,0x07,0x06,0x05,0x04,0x03,0x02,0x01
};

static const uint8_t GOLDEN_TOKEN[RUNTIME_STATE_TOKEN_FRAME_SIZE] = {
    0x4c,0x53,0x52,0x53, 0x05,0x00, 0x04,0x00, 0x30,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00, 0x02,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x08,0x07,0x06,0x05,0x04,0x03,0x02,0x01, 0x04,0x03,0x02,0x01,
    0x44,0x33,0x22,0x11, 0x02,0x02,0x02,0x02,0x01,0x01,0x01,0x01,
    0x02,0x00,0x00,0x00, 0x01,0x00,0x00,0x00,
    0x07,0x00,0x00,0x00,0x00,0x00,0x00,0x00, 0x03,0x00,0x00,0x00,0x00,0x00,0x00,0x00
};

static const uint8_t GOLDEN_TOKEN_RANGE[
    RUNTIME_STATE_TOKEN_CELL_RANGE_FRAME_SIZE] = {
    0x4c,0x53,0x52,0x53, 0x05,0x00, 0x0c,0x00, 0x38,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00, 0x03,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x29,0x00,0x00,0x00,0x00,0x00,0x00,0x00, 0x84,0x03,0x00,0x00,
    0x85,0x03,0x00,0x00, 0xd8,0xd6,0x00,0x00,0x00,0x00,0x00,0x00,
    0x07,0x00,0x00,0x00, 0x02,0x00,0x00,0x00,
    0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x90,0x01,0x00,0x00,0x00,0x00,0x00,0x00,
    0x03,0x00,0x00,0x00, 0x00,0x00,0x00,0x00
};

int main(void) {
    uint8_t frame[RUNTIME_STATE_MAX_FRAME_SIZE];
    uint8_t stream[RUNTIME_STATE_BACKING_FRAME_SIZE + RUNTIME_STATE_TOKEN_FRAME_SIZE
        + RUNTIME_STATE_APPLY_COMPLETE_FRAME_SIZE
        + RUNTIME_STATE_STREAM_VERSION_FRAME_SIZE
        + RUNTIME_STATE_COUNTERS_FRAME_SIZE];
    uint8_t fragment_payload[4] = {0xdeU, 0xadU, 0xbeU, 0xefU};
    uint8_t state_fragment_frame[
        RUNTIME_STATE_HEADER_SIZE
        + RUNTIME_STATE_STATE_UNIT_FRAGMENT_METADATA_SIZE + 4U];
    uint8_t state_fragment_batch_records[
        2U * RUNTIME_STATE_STATE_UNIT_FRAGMENT_METADATA_SIZE + 8U];
    uint8_t state_fragment_batch_frame[
        RUNTIME_STATE_STATE_UNIT_FRAGMENT_BATCH_FRAME_SIZE_MAX];
    uint8_t state_schedule_batch_frame[RUNTIME_STATE_MAX_FRAME_SIZE];
    uint8_t state_seal_frame[RUNTIME_STATE_STATE_UNIT_SEAL_FRAME_SIZE];
    uint8_t bulk_payload[RUNTIME_STATE_BULK_PAYLOAD_BYTES];
    uint8_t bulk_frame[RUNTIME_STATE_STATE_UNIT_BULK_FRAME_SIZE_MAX + 1U];
    uint8_t mixed_stream[sizeof(state_fragment_frame)
        + sizeof(state_seal_frame) + sizeof(stream)];
    struct runtime_state_kv_backing_event backing = {
        UINT64_C(0x0102030405060708), UINT32_C(0x01020304), UINT32_C(0x11223344),
        UINT64_C(0x0101010102020202), UINT64_C(0x1122334455667788), UINT64_C(0x0102030405060708)
    };
    struct runtime_state_token_cell_event token = {
        UINT64_C(0x0102030405060708), UINT32_C(0x01020304), UINT32_C(0x11223344),
        UINT64_C(0x0101010102020202), 2U, 1U, 7U, 3U
    };
    struct runtime_state_token_cell_range_event token_range = {
        .apply_call_id = 41U,
        .tgid = 900U,
        .tid = 901U,
        .monotonic_ns = 55000U,
        .seq_id = 7U,
        .stream_index = 2U,
        .first_token_position = 100U,
        .first_cell_index = 400U,
        .token_count = 3U,
        .reserved = 0U,
    };
    struct runtime_state_apply_complete_event apply_complete = {
        UINT64_C(0x0102030405060708), UINT32_C(0x01020304),
        UINT32_C(0x11223344), UINT64_C(0x0101010102020203)
    };
    struct runtime_state_topology_event topology = {1U, 4242U, 4242U, 90U, 768U, 4U, 4U, 24U, 1U, 1U};
    struct runtime_state_tensor_layout_event tensor = {
        2U, 4242U, 4242U, 91U, RUNTIME_STATE_TENSOR_KIND_V, 23U, 3U, 1U,
        UINT64_C(0x12340000), 16U, 192U, 2U, 8U
    };
    struct runtime_state_terminal_counters terminal = {
        .cutoff_monotonic_ns = 210U,
        .topology_event_count = 0U,
        .tensor_layout_event_count = 0U,
        .kv_backing_event_count = 1U,
        .token_cell_event_count = 1U,
        .apply_complete_event_count = 1U,
        .kernel_event_submit_count = 3U,
        .native_event_receive_count = 3U,
        .native_wire_frame_count = 3U,
        .ringbuf_reserve_fail_count = 0U,
        .user_memory_read_failure_count = 0U,
        .unmatched_call_count = 0U,
        .unsupported_mutation_count = 0U,
        .mapping_mutation_count = 0U,
        .final_inflight_count = 0U,
        .mutation_event_count = 0U,
        .active_emitter_count = 0U,
    };
    struct runtime_state_stream_version_snapshot stream_version = {
        .snapshot_id = 1U,
        .mutation_generation = 7U,
        .backing_generation = 3U,
        .layout_generation = 5U,
        .active_stream_count = 4U,
        .stream_apply_call_ids = {11U, 12U, 0U, 14U},
        .stream_states = {
            RUNTIME_STATE_STREAM_READY,
            RUNTIME_STATE_STREAM_BUSY,
            RUNTIME_STATE_STREAM_EMPTY,
            RUNTIME_STATE_STREAM_READY,
        },
    };
    struct runtime_state_state_unit_fragment fragment = {
        .apply_call_id = 7U,
        .fragment_index = 0U,
        .seq_id = 2U,
        .stream_index = 1U,
        .tensor_kind = RUNTIME_STATE_TENSOR_KIND_K,
        .token_position = 9U,
        .cell_index = 11U,
        .layer = 3U,
        .payload_size = sizeof(fragment_payload),
        .tensor_byte_offset = 0x120U,
        .writer_monotonic_ns = 150U,
        .span_cell_count = 2U,
    };
    struct runtime_state_state_unit_seal state_seal = {
        .apply_call_id = 7U,
        .expected_fragment_count = 1U,
        .expected_token_cell_count = 1U,
        .stream_mask = 2U,
        .final_writer_state = RUNTIME_STATE_WRITER_COMPLETE,
        .layout_generation = 0U,
        .backing_generation = 0U,
        .mutation_generation = 0U,
        .copy_byte_count = sizeof(fragment_payload),
        .reserve_failure_baseline = 0U,
        .reserve_failure_final = 0U,
        .writer_return_monotonic_ns = 151U,
    };
    struct runtime_state_kv_backing_event decoded_backing = {0};
    struct runtime_state_token_cell_event decoded_token = {0};
    struct runtime_state_token_cell_range_event decoded_token_range = {0};
    struct runtime_state_apply_complete_event decoded_apply_complete = {0};
    struct runtime_state_topology_event decoded_topology = {0};
    struct runtime_state_tensor_layout_event decoded_tensor = {0};
    struct runtime_state_terminal_counters decoded_terminal = {0};
    struct runtime_state_stream_version_snapshot decoded_stream_version = {0};
    struct runtime_state_state_unit_fragment decoded_fragment = {0};
    struct runtime_state_state_unit_seal decoded_state_seal = {0};
    const uint8_t *decoded_fragment_payload = NULL;
    const uint8_t *decoded_batch_records = NULL;
    uint32_t decoded_batch_record_count = 0U;
    uint32_t decoded_batch_record_bytes = 0U;
    size_t first_batch_record_size = 0U;
    size_t second_batch_record_size = 0U;
    size_t state_fragment_batch_size = 0U;
    struct runtime_state_kernel_state_unit_fragment_batch_event kernel_batch = {0};
    size_t kernel_batch_size = 0U;
    struct runtime_state_kernel_state_unit_schedule_batch_event
        kernel_schedule_batch = {0};
    struct runtime_state_state_unit_read_plan read_plan = {
        .apply_call_id = 19U,
        .tensor_address = UINT64_C(0x100000),
        .first_token_position = 9U,
        .first_cell_index = 11U,
        .writer_monotonic_ns = 150U,
        .copy_byte_count = 128U,
        .first_fragment_index = 0U,
        .fragment_count = 1U,
        .seq_id = 2U,
        .stream_index = 1U,
        .tensor_kind = RUNTIME_STATE_TENSOR_KIND_K,
        .layer = 0U,
        .expected_n_layer = 3U,
        .token_count = 2U,
        .row_size = 64U,
        .cell_capacity = 32U,
        .element_size = 2U,
        .embedding_width = 32U,
        .chunks_per_dimension = 1U,
        .fragments_per_tensor = 1U,
        .transposed_value = 0U,
        .reserved = 0U,
    };
    struct runtime_state_state_unit_read_plan next_read_plan;
    runtime_state_state_unit_apply_ready apply_ready = {
        .apply_call_id = 19U,
        .expected_fragment_count = 2U,
        .expected_token_cell_count = 2U,
        .stream_mask = 2U,
        .final_writer_state = RUNTIME_STATE_WRITER_COMPLETE,
        .layout_generation = 1U,
        .backing_generation = 1U,
        .mutation_generation = 0U,
        .copy_byte_count = 256U,
        .reserve_failure_baseline = 0U,
        .reserve_failure_final = 0U,
        .writer_return_monotonic_ns = 150U,
    };
    struct runtime_state_state_unit_schedule_batch decoded_schedule_batch = {0};
    const uint8_t *decoded_schedule_payload = NULL;
    size_t state_schedule_batch_size = 0U;
    struct runtime_state_state_unit_read_plan bulk_plan = {
        .apply_call_id = 23U,
        .tensor_address = UINT64_C(0x200000),
        .first_token_position = 100U,
        .first_cell_index = 7U,
        .writer_monotonic_ns = 200U,
        .copy_byte_count = RUNTIME_STATE_BULK_PAYLOAD_BYTES,
        .first_fragment_index = 40U,
        .fragment_count = 16U,
        .seq_id = 3U,
        .stream_index = 2U,
        .tensor_kind = RUNTIME_STATE_TENSOR_KIND_K,
        .layer = 1U,
        .expected_n_layer = 3U,
        .token_count = 64U,
        .row_size = 1024U,
        .cell_capacity = 128U,
        .element_size = 32U,
        .embedding_width = 32U,
        .chunks_per_dimension = 16U,
        .fragments_per_tensor = 16U,
        .transposed_value = 0U,
        .reserved = 0U,
    };
    struct runtime_state_bulk_window bulk_window = {0};
    struct runtime_state_state_unit_bulk bulk = {0};
    struct runtime_state_state_unit_bulk decoded_bulk = {0};
    struct runtime_state_frame_header decoded_bulk_header = {0};
    const uint8_t *decoded_bulk_payload = NULL;
    size_t decoded_bulk_payload_size = 0U;
    size_t bulk_frame_size = 0U;
    uint64_t bulk_prefix_size = 0U;
    size_t index;

    for (index = 0U; index < sizeof(bulk_payload); ++index) {
        bulk_payload[index] = (uint8_t)(index & 0xffU);
    }

    CHECK(RUNTIME_STATE_WIRE_VERSION == 5U);
    CHECK(RUNTIME_STATE_FRAME_STATE_UNIT_BULK == 15U);
    CHECK(RUNTIME_STATE_BULK_PAYLOAD_BYTES == 65536U);
    CHECK(runtime_state_bulk_window_for_plan(
        &bulk_plan, 0U, RUNTIME_STATE_BULK_PAYLOAD_BYTES, &bulk_window));
    CHECK(bulk_window.first_fragment_offset == 0U);
    CHECK(bulk_window.fragment_count == 16U);
    CHECK(bulk_window.payload_offset == 0U);
    CHECK(bulk_window.payload_size == RUNTIME_STATE_BULK_PAYLOAD_BYTES);
    CHECK(bulk_window.flags
        == (RUNTIME_STATE_BULK_FLAG_FIRST | RUNTIME_STATE_BULK_FLAG_FINAL));
    CHECK(runtime_state_bulk_from_plan(&bulk_plan, &bulk_window, &bulk));
    CHECK(runtime_state_encode_state_unit_bulk_frame(
        bulk_frame, sizeof(bulk_frame), &bulk, bulk_payload, 7U,
        &bulk_frame_size));
    CHECK(bulk_frame_size == RUNTIME_STATE_STATE_UNIT_BULK_FRAME_SIZE_MAX);
    CHECK(runtime_state_decode_header(
        bulk_frame, bulk_frame_size, &decoded_bulk_header));
    CHECK(decoded_bulk_header.frame_sequence == 7U);
    CHECK(runtime_state_decode_state_unit_bulk_frame(
        bulk_frame, bulk_frame_size, &decoded_bulk, &decoded_bulk_payload,
        &decoded_bulk_payload_size));
    CHECK(decoded_bulk_payload_size == RUNTIME_STATE_BULK_PAYLOAD_BYTES);
    CHECK(decoded_bulk.frame_first_fragment_index == 40U);
    CHECK(decoded_bulk.frame_fragment_count == 16U);
    CHECK(memcmp(
        decoded_bulk_payload, bulk_payload, decoded_bulk_payload_size) == 0);

    bulk_plan.token_count = 65U;
    bulk_plan.copy_byte_count = 65U * 1024U;
    bulk_plan.chunks_per_dimension = 17U;
    bulk_plan.fragments_per_tensor = 17U;
    bulk_plan.fragment_count = 17U;
    CHECK(runtime_state_bulk_window_for_plan(
        &bulk_plan, 0U, RUNTIME_STATE_BULK_PAYLOAD_BYTES, &bulk_window));
    CHECK(bulk_window.fragment_count == 16U);
    CHECK(bulk_window.flags == RUNTIME_STATE_BULK_FLAG_FIRST);
    CHECK(runtime_state_bulk_window_for_plan(
        &bulk_plan, 16U, RUNTIME_STATE_BULK_PAYLOAD_BYTES, &bulk_window));
    CHECK(bulk_window.fragment_count == 1U);
    CHECK(bulk_window.payload_offset == RUNTIME_STATE_BULK_PAYLOAD_BYTES);
    CHECK(bulk_window.payload_size == 1024U);
    CHECK(bulk_window.flags == RUNTIME_STATE_BULK_FLAG_FINAL);
    CHECK(runtime_state_bulk_from_plan(&bulk_plan, &bulk_window, &bulk));
    CHECK(runtime_state_encode_state_unit_bulk_frame(
        bulk_frame, sizeof(bulk_frame), &bulk,
        bulk_payload, 8U, &bulk_frame_size));
    CHECK(runtime_state_decode_header(
        bulk_frame, bulk_frame_size, &decoded_bulk_header));
    CHECK(decoded_bulk_header.frame_sequence == 8U);
    CHECK(runtime_state_decode_state_unit_bulk_frame(
        bulk_frame, bulk_frame_size, &decoded_bulk, &decoded_bulk_payload,
        &decoded_bulk_payload_size));
    CHECK(decoded_bulk_payload_size == 1024U);
    CHECK(!runtime_state_decode_state_unit_bulk_frame(
        bulk_frame, bulk_frame_size - 1U, &decoded_bulk,
        &decoded_bulk_payload, &decoded_bulk_payload_size));
    bulk_frame[bulk_frame_size] = 0U;
    CHECK(!runtime_state_decode_state_unit_bulk_frame(
        bulk_frame, bulk_frame_size + 1U, &decoded_bulk,
        &decoded_bulk_payload, &decoded_bulk_payload_size));
    bulk.flags = 4U;
    CHECK(!runtime_state_encode_state_unit_bulk_frame(
        bulk_frame, sizeof(bulk_frame), &bulk, bulk_payload, 9U,
        &bulk_frame_size));
    CHECK(runtime_state_bulk_from_plan(&bulk_plan, &bulk_window, &bulk));
    bulk.frame_fragment_count = 0U;
    CHECK(!runtime_state_encode_state_unit_bulk_frame(
        bulk_frame, sizeof(bulk_frame), &bulk, bulk_payload, 9U,
        &bulk_frame_size));
    CHECK(runtime_state_bulk_from_plan(&bulk_plan, &bulk_window, &bulk));
    bulk.frame_payload_offset = UINT64_MAX;
    CHECK(!runtime_state_encode_state_unit_bulk_frame(
        bulk_frame, sizeof(bulk_frame), &bulk, bulk_payload, 9U,
        &bulk_frame_size));

    bulk_plan.tensor_kind = RUNTIME_STATE_TENSOR_KIND_V;
    bulk_plan.token_count = 33U;
    bulk_plan.row_size = 8192U;
    bulk_plan.cell_capacity = 128U;
    bulk_plan.element_size = 2U;
    bulk_plan.embedding_width = 4096U;
    bulk_plan.chunks_per_dimension = 1U;
    bulk_plan.fragments_per_tensor = 4096U;
    bulk_plan.fragment_count = 4096U;
    bulk_plan.copy_byte_count = UINT64_C(270336);
    bulk_plan.transposed_value = 1U;
    CHECK(runtime_state_bulk_prefix_size(&bulk_plan, 0U, &bulk_prefix_size));
    CHECK(bulk_prefix_size == 0U);
    CHECK(runtime_state_bulk_prefix_size(
        &bulk_plan, 992U, &bulk_prefix_size));
    CHECK(bulk_prefix_size == UINT64_C(65472));
    CHECK(runtime_state_bulk_prefix_size(
        &bulk_plan, bulk_plan.fragment_count, &bulk_prefix_size));
    CHECK(bulk_prefix_size == bulk_plan.copy_byte_count);
    CHECK(!runtime_state_bulk_prefix_size(
        &bulk_plan, bulk_plan.fragment_count + 1U, &bulk_prefix_size));
    CHECK(runtime_state_bulk_window_for_plan(
        &bulk_plan, 0U, RUNTIME_STATE_BULK_PAYLOAD_BYTES, &bulk_window));
    CHECK(bulk_window.fragment_count == 992U);
    CHECK(bulk_window.payload_offset == 0U);
    CHECK(bulk_window.payload_size == 65472U);
    CHECK(bulk_window.flags == RUNTIME_STATE_BULK_FLAG_FIRST);

    CHECK(RUNTIME_STATE_MAX_FRAME_SIZE
        == RUNTIME_STATE_STATE_UNIT_BULK_FRAME_SIZE_MAX);
    CHECK(sizeof(struct runtime_state_kernel_state_unit_read_plan_event)
        == 120U);
    CHECK(sizeof(struct runtime_state_kernel_state_unit_apply_ready_event)
        == 88U);
    CHECK(runtime_state_valid_state_unit_read_plan(&read_plan));
    next_read_plan = read_plan;
    next_read_plan.layer = 1U;
    next_read_plan.tensor_address += 2048U;
    next_read_plan.first_fragment_index = 1U;
    CHECK(runtime_state_state_unit_read_plan_follows(
        &read_plan, &next_read_plan));
    next_read_plan.first_fragment_index = 2U;
    CHECK(!runtime_state_state_unit_read_plan_follows(
        &read_plan, &next_read_plan));
    next_read_plan.first_fragment_index = 1U;
    CHECK(runtime_state_valid_state_unit_apply_ready(
        &apply_ready, &next_read_plan, 2U, 256U));
    read_plan.reserved = 1U;
    CHECK(!runtime_state_valid_state_unit_read_plan(&read_plan));
    read_plan.reserved = 0U;

    CHECK(runtime_state_encode_backing_frame(frame, sizeof(frame), &backing, 1U));
    CHECK(memcmp(frame, GOLDEN_BACKING, sizeof(GOLDEN_BACKING)) == 0);
    CHECK(runtime_state_decode_backing_frame(frame, RUNTIME_STATE_BACKING_FRAME_SIZE, &decoded_backing));
    CHECK(decoded_backing.event_id == backing.event_id);
    CHECK(decoded_backing.base_address == backing.base_address);
    CHECK(decoded_backing.size_bytes == backing.size_bytes);
    CHECK(runtime_state_encode_token_frame(frame, sizeof(frame), &token, 2U));
    CHECK(memcmp(frame, GOLDEN_TOKEN, sizeof(GOLDEN_TOKEN)) == 0);
    CHECK(runtime_state_decode_token_frame(frame, RUNTIME_STATE_TOKEN_FRAME_SIZE, &decoded_token));
    CHECK(decoded_token.seq_id == token.seq_id);
    CHECK(decoded_token.stream_index == token.stream_index);
    CHECK(decoded_token.token_position == 7U);
    CHECK(decoded_token.cell_index == 3U);
    CHECK(sizeof(token_range) == 56U);
    CHECK(RUNTIME_STATE_FRAME_TOKEN_CELL_RANGE == 12U);
    CHECK(runtime_state_encode_token_range_frame(
        frame, sizeof(frame), &token_range, 3U));
    CHECK(memcmp(
        frame, GOLDEN_TOKEN_RANGE, sizeof(GOLDEN_TOKEN_RANGE)) == 0);
    CHECK(runtime_state_decode_token_range_frame(
        frame, RUNTIME_STATE_TOKEN_CELL_RANGE_FRAME_SIZE,
        &decoded_token_range));
    CHECK(decoded_token_range.apply_call_id == token_range.apply_call_id);
    CHECK(decoded_token_range.first_token_position == 100U);
    CHECK(decoded_token_range.first_cell_index == 400U);
    CHECK(decoded_token_range.token_count == 3U);
    CHECK(runtime_state_encode_apply_complete_frame(
        frame, sizeof(frame), &apply_complete, 3U));
    CHECK(runtime_state_decode_apply_complete_frame(
        frame, RUNTIME_STATE_APPLY_COMPLETE_FRAME_SIZE,
        &decoded_apply_complete));
    CHECK(decoded_apply_complete.apply_call_id == apply_complete.apply_call_id);
    CHECK(runtime_state_encode_topology_frame(frame, sizeof(frame), &topology, 1U));
    CHECK(runtime_state_decode_topology_frame(frame, RUNTIME_STATE_TOPOLOGY_FRAME_SIZE, &decoded_topology));
    CHECK(runtime_state_load_le32(frame + 48U) == 768U);
    CHECK(runtime_state_load_le32(frame + 60U) == 24U);
    CHECK(decoded_topology.n_layer == 24U);
    CHECK(runtime_state_encode_tensor_frame(frame, sizeof(frame), &tensor, 2U));
    CHECK(runtime_state_decode_tensor_frame(frame, RUNTIME_STATE_TENSOR_FRAME_SIZE, &decoded_tensor));
    CHECK(runtime_state_load_le32(frame + 52U) == 23U);
    CHECK(runtime_state_load_le64(frame + 64U) == UINT64_C(0x12340000));
    CHECK(decoded_tensor.kind == RUNTIME_STATE_TENSOR_KIND_V);
    CHECK(decoded_tensor.stream_index == 3U);
    CHECK(runtime_state_encode_stream_version_frame(
        frame, sizeof(frame), &stream_version, 3U));
    CHECK(runtime_state_decode_stream_version_frame(
        frame, RUNTIME_STATE_STREAM_VERSION_FRAME_SIZE,
        &decoded_stream_version));
    CHECK(decoded_stream_version.snapshot_id == 1U);
    CHECK(decoded_stream_version.active_stream_count == 4U);
    CHECK(decoded_stream_version.stream_apply_call_ids[1] == 12U);
    CHECK(decoded_stream_version.stream_states[1] == RUNTIME_STATE_STREAM_BUSY);
    stream_version.active_stream_count = 5U;
    CHECK(!runtime_state_encode_stream_version_frame(
        frame, sizeof(frame), &stream_version, 3U));
    stream_version.active_stream_count = 4U;
    stream_version.stream_apply_call_ids[4] = 15U;
    stream_version.stream_states[4] = RUNTIME_STATE_STREAM_READY;
    CHECK(!runtime_state_encode_stream_version_frame(
        frame, sizeof(frame), &stream_version, 3U));
    stream_version.stream_apply_call_ids[4] = 0U;
    stream_version.stream_states[4] = RUNTIME_STATE_STREAM_EMPTY;
    CHECK(runtime_state_encode_state_unit_fragment_frame(
        state_fragment_frame, sizeof(state_fragment_frame),
        &fragment, fragment_payload, 1U));
    CHECK(runtime_state_decode_state_unit_fragment_frame(
        state_fragment_frame, sizeof(state_fragment_frame),
        &decoded_fragment, &decoded_fragment_payload));
    CHECK(decoded_fragment.apply_call_id == 7U);
    CHECK(decoded_fragment.payload_size == sizeof(fragment_payload));
    CHECK(decoded_fragment.span_cell_count == 2U);
    CHECK(memcmp(
        decoded_fragment_payload, fragment_payload,
        sizeof(fragment_payload)) == 0);
    CHECK(runtime_state_encode_state_unit_fragment_record(
        state_fragment_batch_records,
        sizeof(state_fragment_batch_records),
        &fragment,
        fragment_payload,
        &first_batch_record_size));
    fragment.fragment_index = 1U;
    fragment.tensor_kind = RUNTIME_STATE_TENSOR_KIND_V;
    CHECK(runtime_state_encode_state_unit_fragment_record(
        state_fragment_batch_records + first_batch_record_size,
        sizeof(state_fragment_batch_records) - first_batch_record_size,
        &fragment,
        fragment_payload,
        &second_batch_record_size));
    CHECK(runtime_state_encode_state_unit_fragment_batch_frame(
        state_fragment_batch_frame,
        sizeof(state_fragment_batch_frame),
        state_fragment_batch_records,
        2U,
        (uint32_t)(first_batch_record_size + second_batch_record_size),
        3U,
        &state_fragment_batch_size));
    CHECK(runtime_state_decode_state_unit_fragment_batch_frame(
        state_fragment_batch_frame,
        state_fragment_batch_size,
        &decoded_batch_records,
        &decoded_batch_record_count,
        &decoded_batch_record_bytes));
    CHECK(decoded_batch_record_count == 2U);
    CHECK(decoded_batch_record_bytes
        == first_batch_record_size + second_batch_record_size);
    CHECK(memcmp(
        decoded_batch_records,
        state_fragment_batch_records,
        decoded_batch_record_bytes) == 0);
    runtime_state_store_le32(
        state_fragment_batch_frame + RUNTIME_STATE_HEADER_SIZE,
        0U);
    CHECK(!runtime_state_decode_state_unit_fragment_batch_frame(
        state_fragment_batch_frame,
        state_fragment_batch_size,
        &decoded_batch_records,
        &decoded_batch_record_count,
        &decoded_batch_record_bytes));
    runtime_state_store_le32(
        state_fragment_batch_frame + RUNTIME_STATE_HEADER_SIZE,
        2U);
    kernel_batch.event_type = RUNTIME_STATE_KERNEL_STATE_UNIT_FRAGMENT_BATCH;
    kernel_batch.record_count = 2U;
    kernel_batch.payload_size = 8U;
    kernel_batch.fragments[0].apply_call_id = 7U;
    kernel_batch.fragments[0].fragment_index = 0U;
    kernel_batch.fragments[0].seq_id = 2U;
    kernel_batch.fragments[0].stream_index = 1U;
    kernel_batch.fragments[0].tensor_kind = RUNTIME_STATE_TENSOR_KIND_K;
    kernel_batch.fragments[0].token_position = 9U;
    kernel_batch.fragments[0].cell_index = 11U;
    kernel_batch.fragments[0].layer = 3U;
    kernel_batch.fragments[0].payload_size = 4U;
    kernel_batch.fragments[0].tensor_byte_offset = 0x120U;
    kernel_batch.fragments[0].writer_monotonic_ns = 150U;
    kernel_batch.fragments[0].span_cell_count = 2U;
    kernel_batch.fragments[1] = kernel_batch.fragments[0];
    kernel_batch.fragments[1].fragment_index = 1U;
    kernel_batch.fragments[1].tensor_kind = RUNTIME_STATE_TENSOR_KIND_V;
    memcpy(kernel_batch.payload, fragment_payload, 4U);
    memcpy(
        kernel_batch.payload + RUNTIME_STATE_MAX_BATCH_FRAGMENT_BYTES,
        fragment_payload,
        4U);
    kernel_batch_size = sizeof(kernel_batch);
    CHECK(runtime_state_encode_kernel_fragment_batch_frame(
        state_fragment_batch_frame,
        sizeof(state_fragment_batch_frame),
        &kernel_batch,
        kernel_batch_size,
        4U,
        &state_fragment_batch_size));
    CHECK(runtime_state_decode_state_unit_fragment_batch_frame(
        state_fragment_batch_frame,
        state_fragment_batch_size,
        &decoded_batch_records,
        &decoded_batch_record_count,
        &decoded_batch_record_bytes));
    CHECK(decoded_batch_record_count == 2U);
    CHECK(decoded_batch_record_bytes
        == 2U * (RUNTIME_STATE_STATE_UNIT_FRAGMENT_METADATA_SIZE + 4U));
    kernel_batch.record_count = 0U;
    CHECK(!runtime_state_encode_kernel_fragment_batch_frame(
        state_fragment_batch_frame,
        sizeof(state_fragment_batch_frame),
        &kernel_batch,
        kernel_batch_size,
        4U,
        &state_fragment_batch_size));
    CHECK(runtime_state_kernel_fragment_batch_can_append(0U, 0U, 4U));
    CHECK(runtime_state_kernel_fragment_batch_can_append(
        0U, 0U, RUNTIME_STATE_MAX_BATCH_FRAGMENT_BYTES));
    CHECK(!runtime_state_kernel_fragment_batch_can_append(
        0U, 0U, RUNTIME_STATE_MAX_BATCH_FRAGMENT_BYTES + 1U));
    CHECK(runtime_state_kernel_fragment_batch_can_append(
        RUNTIME_STATE_MAX_BATCH_FRAGMENTS - 1U,
        RUNTIME_STATE_MAX_BATCH_FRAGMENTS - 1U,
        1U));
    CHECK(!runtime_state_kernel_fragment_batch_can_append(
        RUNTIME_STATE_MAX_BATCH_FRAGMENTS,
        0U,
        4U));
    CHECK(!runtime_state_kernel_fragment_batch_can_append(
        1U,
        RUNTIME_STATE_MAX_FRAGMENT_BYTES,
        1U));
    CHECK(runtime_state_kernel_schedule_batch_can_append(
        RUNTIME_STATE_MAX_BATCH_FRAGMENTS - 1U,
        RUNTIME_STATE_MAX_FRAGMENT_BYTES - 1U,
        1U));
    CHECK(!runtime_state_kernel_schedule_batch_can_append(
        RUNTIME_STATE_MAX_BATCH_FRAGMENTS,
        0U,
        1U));
    kernel_schedule_batch.event_type
        = RUNTIME_STATE_KERNEL_STATE_UNIT_SCHEDULE_BATCH;
    kernel_schedule_batch.apply_call_id = 19U;
    kernel_schedule_batch.first_token_position = 9U;
    kernel_schedule_batch.first_cell_index = 11U;
    kernel_schedule_batch.writer_monotonic_ns = 150U;
    kernel_schedule_batch.record_count = 2U;
    kernel_schedule_batch.payload_size = 8U;
    kernel_schedule_batch.first_fragment_index = 10U;
    kernel_schedule_batch.first_flat_index = 0U;
    kernel_schedule_batch.seq_id = 2U;
    kernel_schedule_batch.stream_index = 1U;
    kernel_schedule_batch.tensor_kind = RUNTIME_STATE_TENSOR_KIND_V;
    kernel_schedule_batch.fragments_per_tensor = 2U;
    kernel_schedule_batch.chunks_per_dimension = 1U;
    kernel_schedule_batch.token_count = 2U;
    kernel_schedule_batch.row_size = 4U;
    kernel_schedule_batch.cell_capacity = 32U;
    kernel_schedule_batch.element_size = 2U;
    kernel_schedule_batch.embedding_width = 2U;
    kernel_schedule_batch.transposed_value = 1U;
    memcpy(kernel_schedule_batch.payload, fragment_payload, 4U);
    memcpy(
        kernel_schedule_batch.payload
            + RUNTIME_STATE_MAX_BATCH_FRAGMENT_BYTES,
        fragment_payload,
        4U);
    CHECK(runtime_state_encode_kernel_schedule_batch_frame(
        state_schedule_batch_frame,
        sizeof(state_schedule_batch_frame),
        &kernel_schedule_batch,
        sizeof(kernel_schedule_batch),
        5U,
        &state_schedule_batch_size));
    CHECK(runtime_state_decode_state_unit_schedule_batch_frame(
        state_schedule_batch_frame,
        state_schedule_batch_size,
        &decoded_schedule_batch,
        &decoded_schedule_payload));
    CHECK(decoded_schedule_batch.record_count == 2U);
    CHECK(decoded_schedule_batch.payload_size == 8U);
    CHECK(decoded_schedule_batch.first_fragment_index == 10U);
    CHECK(decoded_schedule_batch.first_flat_index == 0U);
    CHECK(memcmp(decoded_schedule_payload, fragment_payload, 4U) == 0);
    CHECK(memcmp(
        decoded_schedule_payload + 4U,
        fragment_payload,
        4U) == 0);
    kernel_schedule_batch.fragments_per_tensor = 1U;
    CHECK(!runtime_state_encode_kernel_schedule_batch_frame(
        state_schedule_batch_frame,
        sizeof(state_schedule_batch_frame),
        &kernel_schedule_batch,
        sizeof(kernel_schedule_batch),
        5U,
        &state_schedule_batch_size));
    CHECK(runtime_state_encode_state_unit_seal_frame(
        state_seal_frame, sizeof(state_seal_frame), &state_seal, 2U));
    CHECK(runtime_state_decode_state_unit_seal_frame(
        state_seal_frame, sizeof(state_seal_frame), &decoded_state_seal));
    CHECK(decoded_state_seal.expected_fragment_count == 1U);
    CHECK(runtime_state_tensor_kind_for_ordinal(0U, 4U) == RUNTIME_STATE_TENSOR_KIND_K);
    CHECK(runtime_state_tensor_kind_for_ordinal(1U, 4U) == RUNTIME_STATE_TENSOR_KIND_V);
    CHECK(runtime_state_tensor_kind_for_ordinal(2U, 4U) == RUNTIME_STATE_TENSOR_KIND_K);
    CHECK(runtime_state_tensor_stream_for_ordinal(1U, 4U) == 0U);
    CHECK(runtime_state_tensor_stream_for_ordinal(2U, 4U) == 1U);
    CHECK(runtime_state_tensor_layer_for_ordinal(8U, 4U) == 1U);

    memcpy(stream, GOLDEN_BACKING, sizeof(GOLDEN_BACKING));
    memcpy(stream + sizeof(GOLDEN_BACKING), GOLDEN_TOKEN, sizeof(GOLDEN_TOKEN));
    CHECK(runtime_state_encode_apply_complete_frame(
        stream + sizeof(GOLDEN_BACKING) + sizeof(GOLDEN_TOKEN),
        RUNTIME_STATE_APPLY_COMPLETE_FRAME_SIZE,
        &apply_complete,
        3U));
    CHECK(runtime_state_encode_stream_version_frame(
        stream + sizeof(GOLDEN_BACKING) + sizeof(GOLDEN_TOKEN)
            + RUNTIME_STATE_APPLY_COMPLETE_FRAME_SIZE,
        RUNTIME_STATE_STREAM_VERSION_FRAME_SIZE,
        &stream_version,
        4U));
    CHECK(runtime_state_encode_terminal_frame(
        stream + sizeof(GOLDEN_BACKING) + sizeof(GOLDEN_TOKEN)
            + RUNTIME_STATE_APPLY_COMPLETE_FRAME_SIZE
            + RUNTIME_STATE_STREAM_VERSION_FRAME_SIZE,
        RUNTIME_STATE_COUNTERS_FRAME_SIZE,
        &terminal,
        5U));
    CHECK(runtime_state_decode_terminal_frame(
        stream + sizeof(GOLDEN_BACKING) + sizeof(GOLDEN_TOKEN)
            + RUNTIME_STATE_APPLY_COMPLETE_FRAME_SIZE
            + RUNTIME_STATE_STREAM_VERSION_FRAME_SIZE,
        RUNTIME_STATE_COUNTERS_FRAME_SIZE,
        &decoded_terminal));
    CHECK(decoded_terminal.kernel_event_submit_count == 3U);
    CHECK(runtime_state_validate_stream(stream, sizeof(stream)));
    CHECK(!runtime_state_validate_stream(stream, sizeof(stream) - 1U));
    memcpy(mixed_stream, state_fragment_frame, sizeof(state_fragment_frame));
    memcpy(mixed_stream + sizeof(state_fragment_frame),
        state_seal_frame, sizeof(state_seal_frame));
    memcpy(mixed_stream + sizeof(state_fragment_frame)
        + sizeof(state_seal_frame), stream, sizeof(stream));
    CHECK(runtime_state_validate_stream(mixed_stream, sizeof(mixed_stream)));
    stream[4] = 3U;
    CHECK(!runtime_state_validate_stream(stream, sizeof(stream)));

    puts("runtime-state-wire-check: passed");
    return 0;
}
