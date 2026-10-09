#ifndef LARIS_RUNTIME_STATE_BULK_H
#define LARIS_RUNTIME_STATE_BULK_H

#include "runtime_state_wire.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define RUNTIME_STATE_BULK_FLAG_FIRST UINT32_C(1)
#define RUNTIME_STATE_BULK_FLAG_FINAL UINT32_C(2)
#define RUNTIME_STATE_BULK_FLAG_MASK \
    (RUNTIME_STATE_BULK_FLAG_FIRST | RUNTIME_STATE_BULK_FLAG_FINAL)

struct runtime_state_bulk_window {
    uint32_t first_fragment_offset;
    uint32_t fragment_count;
    uint64_t payload_offset;
    uint32_t payload_size;
    uint32_t flags;
};

struct runtime_state_state_unit_bulk {
    uint64_t apply_call_id;
    uint64_t writer_monotonic_ns;
    uint64_t plan_copy_byte_count;
    uint64_t frame_payload_offset;
    uint64_t first_token_position;
    uint64_t first_cell_index;
    uint32_t first_fragment_index;
    uint32_t fragment_count;
    uint32_t frame_first_fragment_index;
    uint32_t frame_fragment_count;
    uint32_t payload_size;
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
    uint32_t flags;
    uint32_t reserved;
};

_Static_assert(
    sizeof(struct runtime_state_state_unit_bulk)
        == RUNTIME_STATE_STATE_UNIT_BULK_PREFIX_SIZE,
    "state-unit bulk prefix must be 128 bytes");

static inline bool runtime_state_bulk_prefix_size(
    const struct runtime_state_state_unit_read_plan *plan,
    uint32_t fragment_offset,
    uint64_t *payload_size
) {
    uint32_t unit_size;
    uint32_t units_per_fragment;
    uint64_t expected_chunks;
    uint64_t completed_cells;
    uint64_t packed_size;
    if (payload_size == NULL
        || !runtime_state_valid_state_unit_read_plan(plan)
        || fragment_offset > plan->fragment_count) {
        return false;
    }
    unit_size = plan->transposed_value != 0U
        ? plan->element_size : plan->row_size;
    if (unit_size == 0U) {
        return false;
    }
    units_per_fragment = RUNTIME_STATE_MAX_FRAGMENT_BYTES / unit_size;
    if (units_per_fragment == 0U) {
        return false;
    }
    expected_chunks = ((uint64_t)plan->token_count
            + units_per_fragment - 1U)
        / units_per_fragment;
    if (expected_chunks != plan->chunks_per_dimension) {
        return false;
    }
    if (plan->transposed_value != 0U) {
        const uint64_t dimension
            = fragment_offset / plan->chunks_per_dimension;
        const uint64_t chunk
            = fragment_offset % plan->chunks_per_dimension;
        uint64_t cells_in_dimension = chunk * units_per_fragment;
        if (dimension > plan->embedding_width
            || (dimension == plan->embedding_width && chunk != 0U)) {
            return false;
        }
        if (cells_in_dimension > plan->token_count) {
            cells_in_dimension = plan->token_count;
        }
        completed_cells = dimension * plan->token_count
            + cells_in_dimension;
        packed_size = completed_cells * plan->element_size;
    } else {
        completed_cells = (uint64_t)fragment_offset
            * units_per_fragment;
        if (completed_cells > plan->token_count) {
            completed_cells = plan->token_count;
        }
        packed_size = completed_cells * plan->row_size;
    }
    if (packed_size > plan->copy_byte_count
        || (fragment_offset == plan->fragment_count
            && packed_size != plan->copy_byte_count)) {
        return false;
    }
    *payload_size = packed_size;
    return true;
}

static inline bool runtime_state_bulk_fragment_layout(
    const struct runtime_state_state_unit_read_plan *plan,
    uint32_t fragment_offset,
    uint64_t *payload_offset,
    uint32_t *payload_size
) {
    const uint32_t unit_size = plan != NULL && plan->transposed_value != 0U
        ? plan->element_size : (plan != NULL ? plan->row_size : 0U);
    uint32_t units_per_fragment;
    uint32_t chunk_index;
    uint32_t dimension;
    uint64_t cell_offset;
    uint64_t cell_count;
    uint64_t packed_offset;
    uint64_t packed_size;
    if (payload_offset == NULL || payload_size == NULL
        || !runtime_state_valid_state_unit_read_plan(plan)
        || fragment_offset >= plan->fragment_count
        || unit_size == 0U) {
        return false;
    }
    units_per_fragment = RUNTIME_STATE_MAX_FRAGMENT_BYTES / unit_size;
    if (units_per_fragment == 0U) {
        return false;
    }
    chunk_index = fragment_offset % plan->chunks_per_dimension;
    dimension = plan->transposed_value != 0U
        ? fragment_offset / plan->chunks_per_dimension : 0U;
    cell_offset = (uint64_t)chunk_index * units_per_fragment;
    if (cell_offset >= plan->token_count) {
        return false;
    }
    cell_count = (uint64_t)plan->token_count - cell_offset;
    if (cell_count > units_per_fragment) {
        cell_count = units_per_fragment;
    }
    packed_size = cell_count * unit_size;
    if (plan->transposed_value != 0U) {
        packed_offset = (uint64_t)dimension
                * plan->token_count * plan->element_size
            + cell_offset * plan->element_size;
    } else {
        packed_offset = cell_offset * plan->row_size;
    }
    if (packed_size == 0U || packed_size > RUNTIME_STATE_MAX_FRAGMENT_BYTES
        || packed_size > UINT32_MAX
        || packed_offset > plan->copy_byte_count
        || packed_size > plan->copy_byte_count - packed_offset) {
        return false;
    }
    *payload_offset = packed_offset;
    *payload_size = (uint32_t)packed_size;
    return true;
}

static inline bool runtime_state_bulk_window_for_plan(
    const struct runtime_state_state_unit_read_plan *plan,
    uint32_t first_fragment_offset,
    size_t maximum_payload_size,
    struct runtime_state_bulk_window *window
) {
    uint64_t first_payload_offset = 0U;
    uint64_t payload_limit;
    uint64_t end_payload_offset = 0U;
    uint64_t completed_cells;
    uint64_t end_fragment_offset;
    uint32_t unit_size;
    uint32_t units_per_fragment;
    uint32_t count;
    if (window == NULL || !runtime_state_valid_state_unit_read_plan(plan)
        || first_fragment_offset >= plan->fragment_count
        || maximum_payload_size == 0U
        || maximum_payload_size > RUNTIME_STATE_BULK_PAYLOAD_BYTES) {
        return false;
    }
    if (!runtime_state_bulk_prefix_size(
            plan, first_fragment_offset, &first_payload_offset)) {
        return false;
    }
    payload_limit = plan->copy_byte_count - first_payload_offset
            < maximum_payload_size
        ? plan->copy_byte_count
        : first_payload_offset + maximum_payload_size;
    unit_size = plan->transposed_value != 0U
        ? plan->element_size : plan->row_size;
    units_per_fragment = RUNTIME_STATE_MAX_FRAGMENT_BYTES / unit_size;
    if (plan->transposed_value != 0U) {
        completed_cells = payload_limit / plan->element_size;
        if (completed_cells
                >= (uint64_t)plan->token_count * plan->embedding_width) {
            end_fragment_offset = plan->fragment_count;
        } else {
            const uint64_t dimension
                = completed_cells / plan->token_count;
            const uint64_t cells_in_dimension
                = completed_cells % plan->token_count;
            end_fragment_offset
                = dimension * plan->chunks_per_dimension
                + cells_in_dimension / units_per_fragment;
        }
    } else {
        completed_cells = payload_limit / plan->row_size;
        end_fragment_offset = completed_cells >= plan->token_count
            ? plan->fragment_count
            : completed_cells / units_per_fragment;
    }
    if (end_fragment_offset <= first_fragment_offset
        || end_fragment_offset > plan->fragment_count
        || !runtime_state_bulk_prefix_size(
            plan, (uint32_t)end_fragment_offset, &end_payload_offset)
        || end_payload_offset <= first_payload_offset
        || end_payload_offset - first_payload_offset > UINT32_MAX) {
        return false;
    }
    count = (uint32_t)end_fragment_offset - first_fragment_offset;
    window->first_fragment_offset = first_fragment_offset;
    window->fragment_count = count;
    window->payload_offset = first_payload_offset;
    window->payload_size
        = (uint32_t)(end_payload_offset - first_payload_offset);
    window->flags = (first_fragment_offset == 0U
            ? RUNTIME_STATE_BULK_FLAG_FIRST : 0U)
        | (first_fragment_offset + count == plan->fragment_count
            ? RUNTIME_STATE_BULK_FLAG_FINAL : 0U);
    return true;
}

static inline bool runtime_state_bulk_from_plan(
    const struct runtime_state_state_unit_read_plan *plan,
    const struct runtime_state_bulk_window *window,
    struct runtime_state_state_unit_bulk *bulk
) {
    struct runtime_state_bulk_window expected;
    if (window == NULL || bulk == NULL
        || !runtime_state_bulk_window_for_plan(
            plan, window->first_fragment_offset, window->payload_size,
            &expected)
        || memcmp(window, &expected, sizeof(expected)) != 0
        || plan->first_fragment_index
            > UINT32_MAX - window->first_fragment_offset) {
        return false;
    }
    memset(bulk, 0, sizeof(*bulk));
    bulk->apply_call_id = plan->apply_call_id;
    bulk->writer_monotonic_ns = plan->writer_monotonic_ns;
    bulk->plan_copy_byte_count = plan->copy_byte_count;
    bulk->frame_payload_offset = window->payload_offset;
    bulk->first_token_position = plan->first_token_position;
    bulk->first_cell_index = plan->first_cell_index;
    bulk->first_fragment_index = plan->first_fragment_index;
    bulk->fragment_count = plan->fragment_count;
    bulk->frame_first_fragment_index = plan->first_fragment_index
        + window->first_fragment_offset;
    bulk->frame_fragment_count = window->fragment_count;
    bulk->payload_size = window->payload_size;
    bulk->seq_id = plan->seq_id;
    bulk->stream_index = plan->stream_index;
    bulk->tensor_kind = plan->tensor_kind;
    bulk->layer = plan->layer;
    bulk->expected_n_layer = plan->expected_n_layer;
    bulk->token_count = plan->token_count;
    bulk->row_size = plan->row_size;
    bulk->cell_capacity = plan->cell_capacity;
    bulk->element_size = plan->element_size;
    bulk->embedding_width = plan->embedding_width;
    bulk->chunks_per_dimension = plan->chunks_per_dimension;
    bulk->fragments_per_tensor = plan->fragments_per_tensor;
    bulk->transposed_value = plan->transposed_value;
    bulk->flags = window->flags;
    return true;
}

static inline bool runtime_state_valid_state_unit_bulk(
    const struct runtime_state_state_unit_bulk *bulk
) {
    struct runtime_state_state_unit_read_plan plan;
    struct runtime_state_bulk_window expected;
    uint32_t first_fragment_offset;
    if (bulk == NULL || bulk->reserved != 0U
        || bulk->payload_size == 0U
        || bulk->payload_size > RUNTIME_STATE_BULK_PAYLOAD_BYTES
        || (bulk->flags & ~RUNTIME_STATE_BULK_FLAG_MASK) != 0U
        || bulk->frame_fragment_count == 0U
        || bulk->frame_first_fragment_index < bulk->first_fragment_index) {
        return false;
    }
    first_fragment_offset = bulk->frame_first_fragment_index
        - bulk->first_fragment_index;
    if (first_fragment_offset >= bulk->fragment_count
        || bulk->frame_fragment_count
            > bulk->fragment_count - first_fragment_offset
        || bulk->frame_payload_offset > bulk->plan_copy_byte_count
        || bulk->payload_size
            > bulk->plan_copy_byte_count - bulk->frame_payload_offset) {
        return false;
    }
    memset(&plan, 0, sizeof(plan));
    plan.apply_call_id = bulk->apply_call_id;
    plan.tensor_address = 1U;
    plan.first_token_position = bulk->first_token_position;
    plan.first_cell_index = bulk->first_cell_index;
    plan.writer_monotonic_ns = bulk->writer_monotonic_ns;
    plan.copy_byte_count = bulk->plan_copy_byte_count;
    plan.first_fragment_index = bulk->first_fragment_index;
    plan.fragment_count = bulk->fragment_count;
    plan.seq_id = bulk->seq_id;
    plan.stream_index = bulk->stream_index;
    plan.tensor_kind = bulk->tensor_kind;
    plan.layer = bulk->layer;
    plan.expected_n_layer = bulk->expected_n_layer;
    plan.token_count = bulk->token_count;
    plan.row_size = bulk->row_size;
    plan.cell_capacity = bulk->cell_capacity;
    plan.element_size = bulk->element_size;
    plan.embedding_width = bulk->embedding_width;
    plan.chunks_per_dimension = bulk->chunks_per_dimension;
    plan.fragments_per_tensor = bulk->fragments_per_tensor;
    plan.transposed_value = bulk->transposed_value;
    if (!runtime_state_bulk_window_for_plan(
            &plan, first_fragment_offset, bulk->payload_size, &expected)) {
        return false;
    }
    return expected.fragment_count == bulk->frame_fragment_count
        && expected.payload_offset == bulk->frame_payload_offset
        && expected.payload_size == bulk->payload_size
        && expected.flags == bulk->flags;
}

static inline void runtime_state_encode_state_unit_bulk_prefix(
    uint8_t *destination,
    const struct runtime_state_state_unit_bulk *bulk
) {
    runtime_state_store_le64(destination + 0U, bulk->apply_call_id);
    runtime_state_store_le64(destination + 8U, bulk->writer_monotonic_ns);
    runtime_state_store_le64(destination + 16U, bulk->plan_copy_byte_count);
    runtime_state_store_le64(destination + 24U, bulk->frame_payload_offset);
    runtime_state_store_le64(destination + 32U, bulk->first_token_position);
    runtime_state_store_le64(destination + 40U, bulk->first_cell_index);
    runtime_state_store_le32(destination + 48U, bulk->first_fragment_index);
    runtime_state_store_le32(destination + 52U, bulk->fragment_count);
    runtime_state_store_le32(
        destination + 56U, bulk->frame_first_fragment_index);
    runtime_state_store_le32(destination + 60U, bulk->frame_fragment_count);
    runtime_state_store_le32(destination + 64U, bulk->payload_size);
    runtime_state_store_le32(destination + 68U, bulk->seq_id);
    runtime_state_store_le32(destination + 72U, bulk->stream_index);
    runtime_state_store_le32(destination + 76U, bulk->tensor_kind);
    runtime_state_store_le32(destination + 80U, bulk->layer);
    runtime_state_store_le32(destination + 84U, bulk->expected_n_layer);
    runtime_state_store_le32(destination + 88U, bulk->token_count);
    runtime_state_store_le32(destination + 92U, bulk->row_size);
    runtime_state_store_le32(destination + 96U, bulk->cell_capacity);
    runtime_state_store_le32(destination + 100U, bulk->element_size);
    runtime_state_store_le32(destination + 104U, bulk->embedding_width);
    runtime_state_store_le32(destination + 108U, bulk->chunks_per_dimension);
    runtime_state_store_le32(destination + 112U, bulk->fragments_per_tensor);
    runtime_state_store_le32(destination + 116U, bulk->transposed_value);
    runtime_state_store_le32(destination + 120U, bulk->flags);
    runtime_state_store_le32(destination + 124U, bulk->reserved);
}

static inline void runtime_state_decode_state_unit_bulk_prefix(
    const uint8_t *source,
    struct runtime_state_state_unit_bulk *bulk
) {
    bulk->apply_call_id = runtime_state_load_le64(source + 0U);
    bulk->writer_monotonic_ns = runtime_state_load_le64(source + 8U);
    bulk->plan_copy_byte_count = runtime_state_load_le64(source + 16U);
    bulk->frame_payload_offset = runtime_state_load_le64(source + 24U);
    bulk->first_token_position = runtime_state_load_le64(source + 32U);
    bulk->first_cell_index = runtime_state_load_le64(source + 40U);
    bulk->first_fragment_index = runtime_state_load_le32(source + 48U);
    bulk->fragment_count = runtime_state_load_le32(source + 52U);
    bulk->frame_first_fragment_index = runtime_state_load_le32(source + 56U);
    bulk->frame_fragment_count = runtime_state_load_le32(source + 60U);
    bulk->payload_size = runtime_state_load_le32(source + 64U);
    bulk->seq_id = runtime_state_load_le32(source + 68U);
    bulk->stream_index = runtime_state_load_le32(source + 72U);
    bulk->tensor_kind = runtime_state_load_le32(source + 76U);
    bulk->layer = runtime_state_load_le32(source + 80U);
    bulk->expected_n_layer = runtime_state_load_le32(source + 84U);
    bulk->token_count = runtime_state_load_le32(source + 88U);
    bulk->row_size = runtime_state_load_le32(source + 92U);
    bulk->cell_capacity = runtime_state_load_le32(source + 96U);
    bulk->element_size = runtime_state_load_le32(source + 100U);
    bulk->embedding_width = runtime_state_load_le32(source + 104U);
    bulk->chunks_per_dimension = runtime_state_load_le32(source + 108U);
    bulk->fragments_per_tensor = runtime_state_load_le32(source + 112U);
    bulk->transposed_value = runtime_state_load_le32(source + 116U);
    bulk->flags = runtime_state_load_le32(source + 120U);
    bulk->reserved = runtime_state_load_le32(source + 124U);
}

static inline bool runtime_state_encode_state_unit_bulk_frame(
    uint8_t *destination,
    size_t destination_size,
    const struct runtime_state_state_unit_bulk *bulk,
    const uint8_t *payload,
    uint64_t sequence,
    size_t *frame_size
) {
    size_t payload_size;
    if (destination == NULL || payload == NULL || frame_size == NULL
        || !runtime_state_valid_state_unit_bulk(bulk)) {
        return false;
    }
    payload_size = RUNTIME_STATE_STATE_UNIT_BULK_PREFIX_SIZE
        + (size_t)bulk->payload_size;
    if (payload_size > UINT32_MAX
        || destination_size < RUNTIME_STATE_HEADER_SIZE + payload_size
        || !runtime_state_encode_header_for_channel(
            destination, destination_size,
            RUNTIME_STATE_FRAME_STATE_UNIT_BULK, (uint32_t)payload_size,
            RUNTIME_STATE_CHANNEL_STATE_UNIT, sequence)) {
        return false;
    }
    runtime_state_encode_state_unit_bulk_prefix(
        destination + RUNTIME_STATE_HEADER_SIZE, bulk);
    memcpy(
        destination + RUNTIME_STATE_HEADER_SIZE
            + RUNTIME_STATE_STATE_UNIT_BULK_PREFIX_SIZE,
        payload, bulk->payload_size);
    *frame_size = RUNTIME_STATE_HEADER_SIZE + payload_size;
    return true;
}

static inline bool runtime_state_decode_state_unit_bulk_frame(
    const uint8_t *source,
    size_t source_size,
    struct runtime_state_state_unit_bulk *bulk,
    const uint8_t **payload,
    size_t *payload_size
) {
    struct runtime_state_frame_header header;
    const uint8_t *prefix;
    size_t expected_size;
    if (source == NULL || bulk == NULL || payload == NULL
        || payload_size == NULL
        || !runtime_state_decode_header(source, source_size, &header)
        || header.frame_type != RUNTIME_STATE_FRAME_STATE_UNIT_BULK
        || header.channel_id != RUNTIME_STATE_CHANNEL_STATE_UNIT
        || header.payload_length <= RUNTIME_STATE_STATE_UNIT_BULK_PREFIX_SIZE) {
        return false;
    }
    prefix = source + RUNTIME_STATE_HEADER_SIZE;
    runtime_state_decode_state_unit_bulk_prefix(prefix, bulk);
    expected_size = RUNTIME_STATE_HEADER_SIZE
        + RUNTIME_STATE_STATE_UNIT_BULK_PREFIX_SIZE
        + (size_t)bulk->payload_size;
    if (!runtime_state_valid_state_unit_bulk(bulk)
        || header.payload_length
            != RUNTIME_STATE_STATE_UNIT_BULK_PREFIX_SIZE + bulk->payload_size
        || source_size != expected_size) {
        return false;
    }
    *payload = prefix + RUNTIME_STATE_STATE_UNIT_BULK_PREFIX_SIZE;
    *payload_size = bulk->payload_size;
    return true;
}

#endif
