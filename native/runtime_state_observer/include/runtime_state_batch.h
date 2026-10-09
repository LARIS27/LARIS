#ifndef LARIS_RUNTIME_STATE_BATCH_H
#define LARIS_RUNTIME_STATE_BATCH_H

#include "runtime_state_bpf_shared.h"
#include "runtime_state_wire.h"

static inline bool runtime_state_encode_kernel_fragment_batch_frame(
    uint8_t *destination,
    size_t destination_size,
    const struct runtime_state_kernel_state_unit_fragment_batch_event *batch,
    size_t batch_size,
    uint64_t sequence,
    size_t *frame_size
) {
    uint8_t *batch_payload;
    uint8_t *record_destination;
    size_t canonical_record_bytes;
    size_t payload_offset = 0U;
    size_t payload_slot_offset;
    size_t record_offset = 0U;
    uint32_t index;
    if (destination == NULL || batch == NULL || frame_size == NULL
        || batch->event_type
            != RUNTIME_STATE_KERNEL_STATE_UNIT_FRAGMENT_BATCH
        || batch->reserved != 0U
        || batch->record_count == 0U
        || batch->record_count > RUNTIME_STATE_MAX_BATCH_FRAGMENTS
        || batch->payload_size == 0U
        || batch->payload_size > RUNTIME_STATE_MAX_FRAGMENT_BYTES
        || batch_size != sizeof(*batch)) {
        return false;
    }
    canonical_record_bytes
        = (size_t)batch->record_count
            * RUNTIME_STATE_STATE_UNIT_FRAGMENT_METADATA_SIZE
        + batch->payload_size;
    if (canonical_record_bytes
            > RUNTIME_STATE_STATE_UNIT_FRAGMENT_METADATA_SIZE
                + RUNTIME_STATE_MAX_FRAGMENT_BYTES
        || destination_size
            < RUNTIME_STATE_HEADER_SIZE
                + RUNTIME_STATE_STATE_UNIT_FRAGMENT_BATCH_PREFIX_SIZE
                + canonical_record_bytes
        || !runtime_state_encode_header_for_channel(
            destination, destination_size,
            RUNTIME_STATE_FRAME_STATE_UNIT_FRAGMENT_BATCH,
            (uint32_t)(
                RUNTIME_STATE_STATE_UNIT_FRAGMENT_BATCH_PREFIX_SIZE
                + canonical_record_bytes),
            RUNTIME_STATE_CHANNEL_STATE_UNIT,
            sequence)) {
        return false;
    }
    batch_payload = destination + RUNTIME_STATE_HEADER_SIZE;
    runtime_state_store_le32(
        batch_payload + 0U, batch->record_count);
    runtime_state_store_le32(
        batch_payload + 4U, (uint32_t)canonical_record_bytes);
    record_destination = batch_payload
        + RUNTIME_STATE_STATE_UNIT_FRAGMENT_BATCH_PREFIX_SIZE;
    for (index = 0U; index < batch->record_count; ++index) {
        const struct runtime_state_kernel_state_unit_fragment_metadata *source
            = &batch->fragments[index];
        struct runtime_state_state_unit_fragment fragment;
        size_t encoded_size = 0U;
        if (source->payload_size == 0U
            || source->payload_size
                > RUNTIME_STATE_MAX_BATCH_FRAGMENT_BYTES
            || source->payload_size
                > batch->payload_size - payload_offset) {
            return false;
        }
        payload_slot_offset = (size_t)index
            * RUNTIME_STATE_MAX_BATCH_FRAGMENT_BYTES;
        memcpy(&fragment, source, sizeof(fragment));
        if (!runtime_state_encode_state_unit_fragment_record(
                record_destination + record_offset,
                canonical_record_bytes - record_offset,
                &fragment,
                batch->payload + payload_slot_offset,
                &encoded_size)) {
            return false;
        }
        payload_offset += source->payload_size;
        record_offset += encoded_size;
    }
    if (payload_offset != batch->payload_size
        || record_offset != canonical_record_bytes) {
        return false;
    }
    *frame_size = RUNTIME_STATE_HEADER_SIZE
        + RUNTIME_STATE_STATE_UNIT_FRAGMENT_BATCH_PREFIX_SIZE
        + canonical_record_bytes;
    return true;
}

static inline bool runtime_state_schedule_batch_record_size(
    const struct runtime_state_kernel_state_unit_schedule_batch_event *batch,
    uint32_t record_index,
    uint32_t *record_payload_size
) {
    uint32_t flat_index;
    uint32_t fragment_in_tensor;
    uint32_t chunk_index;
    uint32_t units_per_fragment;
    uint32_t cell_offset;
    uint32_t cell_count;
    if (batch == NULL || record_payload_size == NULL
        || record_index >= batch->record_count
        || batch->element_size == 0U
        || batch->element_size > RUNTIME_STATE_MAX_FRAGMENT_BYTES) {
        return false;
    }
    flat_index = batch->first_flat_index + record_index;
    fragment_in_tensor = flat_index % batch->fragments_per_tensor;
    chunk_index = fragment_in_tensor % batch->chunks_per_dimension;
    units_per_fragment = RUNTIME_STATE_MAX_FRAGMENT_BYTES
        / batch->element_size;
    if (units_per_fragment == 0U
        || chunk_index > UINT32_MAX / units_per_fragment) {
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
    *record_payload_size = cell_count * batch->element_size;
    return *record_payload_size != 0U
        && *record_payload_size <= RUNTIME_STATE_MAX_BATCH_FRAGMENT_BYTES;
}

static inline bool runtime_state_valid_kernel_schedule_batch(
    const struct runtime_state_kernel_state_unit_schedule_batch_event *batch,
    size_t batch_size
) {
    uint64_t row_size;
    uint64_t fragments_per_tensor;
    uint64_t payload_size = 0U;
    uint32_t record_index;
    if (batch == NULL || batch_size != sizeof(*batch)
        || batch->event_type
            != RUNTIME_STATE_KERNEL_STATE_UNIT_SCHEDULE_BATCH
        || batch->reserved != 0U || batch->schedule_reserved != 0U
        || batch->record_count == 0U
        || batch->record_count > RUNTIME_STATE_MAX_BATCH_FRAGMENTS
        || batch->payload_size == 0U
        || batch->payload_size > RUNTIME_STATE_MAX_FRAGMENT_BYTES
        || batch->apply_call_id == 0U
        || batch->stream_index >= RUNTIME_STATE_MAX_APPLY_STREAMS
        || batch->tensor_kind != RUNTIME_STATE_TENSOR_KIND_V
        || batch->fragments_per_tensor == 0U
        || batch->chunks_per_dimension == 0U
        || batch->token_count == 0U
        || batch->row_size == 0U || batch->cell_capacity == 0U
        || batch->element_size == 0U || batch->embedding_width == 0U
        || batch->transposed_value != 1U
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
    for (record_index = 0U;
            record_index < batch->record_count;
            ++record_index) {
        uint32_t record_payload_size = 0U;
        if (!runtime_state_schedule_batch_record_size(
                batch, record_index, &record_payload_size)) {
            return false;
        }
        payload_size += record_payload_size;
    }
    return payload_size == batch->payload_size;
}

static inline bool runtime_state_encode_kernel_schedule_batch_frame(
    uint8_t *destination,
    size_t destination_size,
    const struct runtime_state_kernel_state_unit_schedule_batch_event *batch,
    size_t batch_size,
    uint64_t sequence,
    size_t *frame_size
) {
    uint8_t *payload;
    size_t packed_payload_offset = 0U;
    uint32_t record_index;
    const size_t wire_payload_size
        = RUNTIME_STATE_STATE_UNIT_SCHEDULE_BATCH_PREFIX_SIZE
            + (batch == NULL ? 0U : (size_t)batch->payload_size);
    if (frame_size == NULL
        || !runtime_state_valid_kernel_schedule_batch(batch, batch_size)
        || destination == NULL
        || destination_size < RUNTIME_STATE_HEADER_SIZE + wire_payload_size
        || !runtime_state_encode_header_for_channel(
            destination, destination_size,
            RUNTIME_STATE_FRAME_STATE_UNIT_SCHEDULE_BATCH,
            (uint32_t)wire_payload_size,
            RUNTIME_STATE_CHANNEL_STATE_UNIT,
            sequence)) {
        return false;
    }
    payload = destination + RUNTIME_STATE_HEADER_SIZE;
    runtime_state_store_le32(payload + 0U, batch->record_count);
    runtime_state_store_le32(payload + 4U, batch->payload_size);
    runtime_state_store_le64(payload + 8U, batch->apply_call_id);
    runtime_state_store_le32(payload + 16U, batch->first_fragment_index);
    runtime_state_store_le32(payload + 20U, batch->first_flat_index);
    runtime_state_store_le32(payload + 24U, batch->seq_id);
    runtime_state_store_le32(payload + 28U, batch->stream_index);
    runtime_state_store_le32(payload + 32U, batch->tensor_kind);
    runtime_state_store_le32(payload + 36U, batch->fragments_per_tensor);
    runtime_state_store_le64(payload + 40U, batch->first_token_position);
    runtime_state_store_le64(payload + 48U, batch->first_cell_index);
    runtime_state_store_le32(payload + 56U, batch->chunks_per_dimension);
    runtime_state_store_le32(payload + 60U, batch->token_count);
    runtime_state_store_le32(payload + 64U, batch->row_size);
    runtime_state_store_le32(payload + 68U, batch->cell_capacity);
    runtime_state_store_le32(payload + 72U, batch->element_size);
    runtime_state_store_le32(payload + 76U, batch->embedding_width);
    runtime_state_store_le32(payload + 80U, batch->transposed_value);
    runtime_state_store_le32(payload + 84U, batch->schedule_reserved);
    runtime_state_store_le64(payload + 88U, batch->writer_monotonic_ns);
    for (record_index = 0U;
            record_index < batch->record_count;
            ++record_index) {
        uint32_t record_payload_size = 0U;
        const size_t slot_offset = (size_t)record_index
            * RUNTIME_STATE_MAX_BATCH_FRAGMENT_BYTES;
        if (!runtime_state_schedule_batch_record_size(
                batch, record_index, &record_payload_size)) {
            return false;
        }
        memcpy(
            payload + RUNTIME_STATE_STATE_UNIT_SCHEDULE_BATCH_PREFIX_SIZE
                + packed_payload_offset,
            batch->payload + slot_offset,
            record_payload_size);
        packed_payload_offset += record_payload_size;
    }
    if (packed_payload_offset != batch->payload_size) {
        return false;
    }
    *frame_size = RUNTIME_STATE_HEADER_SIZE + wire_payload_size;
    return true;
}

#endif
