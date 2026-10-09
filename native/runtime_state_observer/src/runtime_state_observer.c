#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "runtime_state_bpf_shared.h"
#include "runtime_state_batch.h"
#include "runtime_state_bulk.h"
#include "runtime_state_wire.h"

#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <limits.h>
#include <poll.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/uio.h>

#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#define BASE_PROBE_COUNT 7U
#define METADATA_BASE_LINK_COUNT 13U
#define WRITER_BASE_LINK_COUNT 15U
#define MAX_MUTATION_PROBES 2U
#define MAX_LINK_COUNT (WRITER_BASE_LINK_COUNT + MAX_MUTATION_PROBES)
#define MAX_PROBE_IDENTITIES 14U
#define MAX_PROBE_IDENTITY_LENGTH 994U
#define RUNTIME_STATE_DRAIN_TIMEOUT_NS UINT64_C(10000000000)
#define RUNTIME_STATE_DRAIN_STALL_DIAGNOSTIC_NS UINT64_C(1000000000)
#define RUNTIME_STATE_PAYLOAD_RING_BYTES_PER_STREAM (128U * 1024U * 1024U)
#define RUNTIME_STATE_WRITER_STDOUT_BUFFER_BYTES (1024U * 1024U)
#define RUNTIME_STATE_GATHER_IOV_MAX 1024U
#define RUNTIME_STATE_GATHER_MAX_PLANS RUNTIME_STATE_MAX_TENSOR_EVENTS
#define RUNTIME_STATE_GATHER_MAX_PENDING 64U
#define RUNTIME_STATE_STOP_TIMEOUT_NS UINT64_C(2000000000)

struct observer_probe_target {
    const char *binary;
    uint64_t offset;
};

struct observer_base_link {
    const char *program_name;
    size_t probe_index;
    bool return_probe;
};

struct observer_mutation_target {
    const char *binary;
    uint64_t offset;
    uint64_t mutation_kind;
};

enum observer_capture_mode {
    OBSERVER_CAPTURE_METADATA = RUNTIME_STATE_CAPTURE_METADATA,
    OBSERVER_CAPTURE_WRITER_BOUNDARY = RUNTIME_STATE_CAPTURE_WRITER_BOUNDARY,
    OBSERVER_CAPTURE_COMPACT_METADATA = RUNTIME_STATE_CAPTURE_COMPACT_METADATA,
    OBSERVER_CAPTURE_GATHER_WRITER_BOUNDARY =
        RUNTIME_STATE_CAPTURE_GATHER_WRITER_BOUNDARY,
};

enum gather_failure_stage {
    GATHER_FAILURE_NONE = 0,
    GATHER_FAILURE_PLAN,
    GATHER_FAILURE_READY,
    GATHER_FAILURE_STOP,
    GATHER_FAILURE_MEMORY_READ,
    GATHER_FAILURE_FRAME_EMIT,
    GATHER_FAILURE_SEAL_EMIT,
    GATHER_FAILURE_RESUME,
};

struct observer_options {
    uint32_t target_tgid;
    const char *bpf_object;
    uint32_t expected_n_stream;
    uint32_t expected_n_seq_max;
    uint32_t expected_n_layer;
    struct observer_probe_target probes[BASE_PROBE_COUNT];
    struct observer_mutation_target mutations[MAX_MUTATION_PROBES];
    size_t mutation_count;
    const char *probe_identities[MAX_PROBE_IDENTITIES];
    size_t probe_identity_count;
    uint32_t mutation_mask;
    uint32_t tensor_data_offset;
    uint32_t tensor_type_offset;
    enum observer_capture_mode capture_mode;
    bool offsets_complete;
};

struct pending_capture {
    uint64_t apply_call_id;
    struct runtime_state_state_unit_read_plan *read_plans;
    size_t read_plan_count;
    uint64_t planned_copy_byte_count;
};

struct gathered_capture {
    runtime_state_state_unit_apply_ready ready;
    struct runtime_state_state_unit_read_plan *read_plans;
    size_t read_plan_count;
    uint8_t *payload;
    size_t payload_size;
};

struct collector_state {
    uint64_t next_metadata_sequence;
    uint64_t next_state_unit_sequence;
    uint64_t next_snapshot_id;
    uint64_t native_event_receive_count;
    uint64_t native_wire_frame_count;
    uint64_t topology_event_count;
    uint64_t tensor_layout_event_count;
    uint64_t kv_backing_event_count;
    uint64_t mutation_event_count;
    uint64_t native_read_plan_count;
    uint64_t native_read_syscall_count;
    uint64_t native_copied_bytes;
    uint64_t native_bulk_frame_count;
    uint64_t native_bulk_payload_bytes;
    uint64_t native_logical_fragment_count;
    uint64_t native_logical_bytes;
    uint64_t native_encode_write_ns;
    uint64_t capture_boundary_hold_ns;
    uint64_t capture_boundary_total_ns;
    uint64_t target_stop_started_ns;
    uint64_t target_boundary_started_ns;
    uint32_t active_stream_count;
    uint64_t expected_tensor_count;
    struct pending_capture pending_captures[
        RUNTIME_STATE_GATHER_MAX_PENDING];
    size_t pending_capture_count;
    struct gathered_capture gathered_captures[
        RUNTIME_STATE_GATHER_MAX_PENDING];
    size_t gathered_capture_count;
    size_t read_plan_capacity;
    pid_t target_tgid;
    enum gather_failure_stage gather_failure_stage;
    int gather_failure_errno;
    size_t gather_failure_pending_count;
    bool target_stopped;
    bool ready_reported;
    bool failed;
};

static volatile sig_atomic_t stop_requested = 0;
static char writer_stdout_buffer[RUNTIME_STATE_WRITER_STDOUT_BUFFER_BYTES];

static bool parse_u64(const char *text, uint64_t *result);
static uint64_t monotonic_ns(void);
static bool accumulate_u64(uint64_t *total, uint64_t value);

static void handle_signal(int signal_number) {
    (void)signal_number;
    stop_requested = 1;
}

static void usage(const char *program) {
    fprintf(stderr,
        "usage: %s --target-tgid PID --bpf-object PATH "
        "--expected-n-stream N --expected-n-seq-max N --expected-n-layer N "
        "--constructor-binary PATH --n-pos-per-embd-binary PATH "
        "--view-binary PATH --buffer-base-binary PATH "
        "--buffer-size-binary PATH --apply-binary PATH "
        "--constructor-offset N --n-pos-per-embd-offset N --view-offset N "
        "--buffer-base-offset N "
        "--buffer-size-offset N --apply-offset N --tensor-data-offset N "
        "--tensor-type-offset N [--capture-mode metadata|compact-metadata|writer-boundary|gather-writer-boundary "
        "--process-ubatch-offset N --process-ubatch-binary PATH] "
        "--mutation-probe KIND|PATH|OFFSET "
        "--probe-identity JSON ...\n",
        program);
}

static bool parse_mutation_target(
    char *text,
    struct observer_mutation_target *target
) {
    char *first;
    char *last;
    uint64_t kind;
    if (text == NULL || target == NULL) return false;
    first = strchr(text, '|');
    last = strrchr(text, '|');
    if (first == NULL || last == first || first == text || last[1] == '\0') {
        return false;
    }
    *first = '\0';
    *last = '\0';
    if (!parse_u64(text, &kind)
        || kind < RUNTIME_STATE_MUTATION_CACHE_SHIFT
        || kind > RUNTIME_STATE_MUTATION_SPECULATIVE_CONTEXT
        || !parse_u64(last + 1, &target->offset)
        || first[1] != '/') {
        return false;
    }
    target->mutation_kind = kind;
    target->binary = first + 1;
    return true;
}

static bool parse_u64(const char *text, uint64_t *result) {
    char *end = NULL;
    unsigned long long value;
    if (text == NULL || *text == '\0' || result == NULL || *text == '-') {
        return false;
    }
    errno = 0;
    value = strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0') {
        return false;
    }
    *result = (uint64_t)value;
    return true;
}

static bool parse_u32(const char *text, uint32_t *result) {
    uint64_t value;
    if (!parse_u64(text, &value) || value == 0U || value > UINT32_MAX) {
        return false;
    }
    *result = (uint32_t)value;
    return true;
}

static bool parse_offset_u32(const char *text, uint32_t *result) {
    uint64_t value;
    if (!parse_u64(text, &value) || value > UINT32_MAX) {
        return false;
    }
    *result = (uint32_t)value;
    return true;
}

static bool parse_options(int argc, char **argv, struct observer_options *options) {
    enum {
        OPT_CONSTRUCTOR = 1000,
        OPT_EXPECTED_N_STREAM,
        OPT_EXPECTED_N_SEQ_MAX,
        OPT_EXPECTED_N_LAYER,
        OPT_N_POS_PER_EMBD,
        OPT_VIEW,
        OPT_BUFFER_BASE,
        OPT_BUFFER_SIZE,
        OPT_APPLY,
        OPT_TENSOR_DATA,
        OPT_TENSOR_TYPE,
        OPT_CONSTRUCTOR_BINARY,
        OPT_N_POS_PER_EMBD_BINARY,
        OPT_VIEW_BINARY,
        OPT_BUFFER_BASE_BINARY,
        OPT_BUFFER_SIZE_BINARY,
        OPT_APPLY_BINARY,
        OPT_PROCESS_UBATCH,
        OPT_PROCESS_UBATCH_BINARY,
        OPT_CAPTURE_MODE,
        OPT_MUTATION_PROBE,
        OPT_PROBE_IDENTITY,
    };
    static const struct option long_options[] = {
        {"target-tgid", required_argument, NULL, 'p'},
        {"bpf-object", required_argument, NULL, 'o'},
        {"expected-n-stream", required_argument, NULL, OPT_EXPECTED_N_STREAM},
        {"expected-n-seq-max", required_argument, NULL, OPT_EXPECTED_N_SEQ_MAX},
        {"expected-n-layer", required_argument, NULL, OPT_EXPECTED_N_LAYER},
        {"constructor-offset", required_argument, NULL, OPT_CONSTRUCTOR},
        {"n-pos-per-embd-offset", required_argument, NULL, OPT_N_POS_PER_EMBD},
        {"view-offset", required_argument, NULL, OPT_VIEW},
        {"buffer-base-offset", required_argument, NULL, OPT_BUFFER_BASE},
        {"buffer-size-offset", required_argument, NULL, OPT_BUFFER_SIZE},
        {"apply-offset", required_argument, NULL, OPT_APPLY},
        {"tensor-data-offset", required_argument, NULL, OPT_TENSOR_DATA},
        {"tensor-type-offset", required_argument, NULL, OPT_TENSOR_TYPE},
        {"constructor-binary", required_argument, NULL, OPT_CONSTRUCTOR_BINARY},
        {"n-pos-per-embd-binary", required_argument, NULL, OPT_N_POS_PER_EMBD_BINARY},
        {"view-binary", required_argument, NULL, OPT_VIEW_BINARY},
        {"buffer-base-binary", required_argument, NULL, OPT_BUFFER_BASE_BINARY},
        {"buffer-size-binary", required_argument, NULL, OPT_BUFFER_SIZE_BINARY},
        {"apply-binary", required_argument, NULL, OPT_APPLY_BINARY},
        {"process-ubatch-offset", required_argument, NULL, OPT_PROCESS_UBATCH},
        {"process-ubatch-binary", required_argument, NULL, OPT_PROCESS_UBATCH_BINARY},
        {"capture-mode", required_argument, NULL, OPT_CAPTURE_MODE},
        {"mutation-probe", required_argument, NULL, OPT_MUTATION_PROBE},
        {"probe-identity", required_argument, NULL, OPT_PROBE_IDENTITY},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0},
    };
    bool constructor_set = false;
    bool expected_n_stream_set = false;
    bool expected_n_seq_max_set = false;
    bool expected_n_layer_set = false;
    bool n_pos_per_embd_set = false;
    bool view_set = false;
    bool buffer_base_set = false;
    bool buffer_size_set = false;
    bool apply_set = false;
    bool process_ubatch_set = false;
    bool tensor_data_set = false;
    bool tensor_type_set = false;
    bool binary_set[BASE_PROBE_COUNT] = {false};
    int option;
    memset(options, 0, sizeof(*options));
    while ((option = getopt_long(argc, argv, "p:o:h", long_options, NULL)) != -1) {
        switch (option) {
            case 'p': if (!parse_u32(optarg, &options->target_tgid)) return false; break;
            case 'o': options->bpf_object = optarg; break;
            case OPT_CONSTRUCTOR: constructor_set = parse_u64(optarg, &options->probes[0].offset); break;
            case OPT_EXPECTED_N_STREAM: expected_n_stream_set = parse_u32(optarg, &options->expected_n_stream); break;
            case OPT_EXPECTED_N_SEQ_MAX: expected_n_seq_max_set = parse_u32(optarg, &options->expected_n_seq_max); break;
            case OPT_EXPECTED_N_LAYER: expected_n_layer_set = parse_offset_u32(optarg, &options->expected_n_layer); break;
            case OPT_N_POS_PER_EMBD: n_pos_per_embd_set = parse_u64(optarg, &options->probes[1].offset); break;
            case OPT_VIEW: view_set = parse_u64(optarg, &options->probes[2].offset); break;
            case OPT_BUFFER_BASE: buffer_base_set = parse_u64(optarg, &options->probes[3].offset); break;
            case OPT_BUFFER_SIZE: buffer_size_set = parse_u64(optarg, &options->probes[4].offset); break;
            case OPT_APPLY: apply_set = parse_u64(optarg, &options->probes[5].offset); break;
            case OPT_TENSOR_DATA: tensor_data_set = parse_offset_u32(optarg, &options->tensor_data_offset); break;
            case OPT_TENSOR_TYPE: tensor_type_set = parse_offset_u32(optarg, &options->tensor_type_offset); break;
            case OPT_CONSTRUCTOR_BINARY: options->probes[0].binary = optarg; binary_set[0] = true; break;
            case OPT_N_POS_PER_EMBD_BINARY: options->probes[1].binary = optarg; binary_set[1] = true; break;
            case OPT_VIEW_BINARY: options->probes[2].binary = optarg; binary_set[2] = true; break;
            case OPT_BUFFER_BASE_BINARY: options->probes[3].binary = optarg; binary_set[3] = true; break;
            case OPT_BUFFER_SIZE_BINARY: options->probes[4].binary = optarg; binary_set[4] = true; break;
            case OPT_APPLY_BINARY: options->probes[5].binary = optarg; binary_set[5] = true; break;
            case OPT_PROCESS_UBATCH: process_ubatch_set = parse_u64(optarg, &options->probes[6].offset); break;
            case OPT_PROCESS_UBATCH_BINARY: options->probes[6].binary = optarg; binary_set[6] = true; break;
            case OPT_CAPTURE_MODE:
                if (strcmp(optarg, "writer-boundary") == 0) {
                    options->capture_mode = OBSERVER_CAPTURE_WRITER_BOUNDARY;
                } else if (strcmp(optarg, "gather-writer-boundary") == 0) {
                    options->capture_mode =
                        OBSERVER_CAPTURE_GATHER_WRITER_BOUNDARY;
                } else if (strcmp(optarg, "compact-metadata") == 0) {
                    options->capture_mode = OBSERVER_CAPTURE_COMPACT_METADATA;
                } else if (strcmp(optarg, "metadata") == 0) {
                    options->capture_mode = OBSERVER_CAPTURE_METADATA;
                } else {
                    return false;
                }
                break;
            case OPT_MUTATION_PROBE:
                if (options->mutation_count >= MAX_MUTATION_PROBES
                    || !parse_mutation_target(
                        optarg, &options->mutations[options->mutation_count])) return false;
                {
                    const uint32_t bit = 1U << (
                        options->mutations[options->mutation_count].mutation_kind - 1U);
                    if (bit == (1U << (RUNTIME_STATE_MUTATION_SEQUENCE_STREAM_REMAP - 1U))
                        || (options->mutation_mask & bit) != 0U) return false;
                    options->mutation_mask |= bit;
                }
                options->mutation_count += 1U;
                break;
            case OPT_PROBE_IDENTITY:
                if (options->probe_identity_count >= MAX_PROBE_IDENTITIES
                    || optarg == NULL
                    || optarg[0] != '{'
                    || strlen(optarg) > MAX_PROBE_IDENTITY_LENGTH
                    || strchr(optarg, '\n') != NULL
                    || strchr(optarg, '\r') != NULL) {
                    return false;
                }
                options->probe_identities[options->probe_identity_count] = optarg;
                options->probe_identity_count += 1U;
                break;
            case 'h': usage(argv[0]); exit(0);
            default: return false;
        }
    }
    options->offsets_complete = constructor_set && expected_n_stream_set
        && expected_n_seq_max_set && expected_n_layer_set && n_pos_per_embd_set
        && view_set && buffer_base_set
        && buffer_size_set && apply_set && tensor_data_set && tensor_type_set;
    return optind == argc && options->target_tgid != 0U
        && options->bpf_object != NULL && options->offsets_complete
        && binary_set[0] && binary_set[1] && binary_set[2]
        && binary_set[3] && binary_set[4] && binary_set[5]
        && ((options->capture_mode != OBSERVER_CAPTURE_WRITER_BOUNDARY
                && options->capture_mode
                    != OBSERVER_CAPTURE_GATHER_WRITER_BOUNDARY)
            || (process_ubatch_set && binary_set[6]))
        && ((options->capture_mode == OBSERVER_CAPTURE_WRITER_BOUNDARY
                || options->capture_mode
                    == OBSERVER_CAPTURE_GATHER_WRITER_BOUNDARY)
            || (!process_ubatch_set && !binary_set[6]))
        && options->mutation_count == MAX_MUTATION_PROBES
        && options->mutation_mask == UINT32_C(0x00000030)
        && options->expected_n_stream == options->expected_n_seq_max
        && (options->expected_n_stream == 4U
            || options->expected_n_stream == 8U)
        && options->expected_n_layer <= RUNTIME_STATE_MAX_TENSOR_EVENTS
            / (2U * options->expected_n_stream);
}

static bool write_frame(const uint8_t *frame, size_t frame_size) {
    return fwrite(frame, 1U, frame_size, stdout) == frame_size;
}

static void report_ready_if_complete(struct collector_state *state) {
    if (!state->ready_reported
        && state->topology_event_count == 1U
        && state->expected_tensor_count != 0U
        && state->tensor_layout_event_count == state->expected_tensor_count
        && state->kv_backing_event_count != 0U) {
        fprintf(stderr,
            "READY runtime_state topology=%" PRIu64
            " tensors=%" PRIu64 " backing=%" PRIu64 "\n",
            state->topology_event_count,
            state->tensor_layout_event_count,
            state->kv_backing_event_count);
        (void)fflush(stderr);
        state->ready_reported = true;
    }
}

static int collect_event(void *context, void *data, size_t size) {
    struct collector_state *state = context;
    const struct runtime_state_kernel_event *event = data;
    uint8_t frame[RUNTIME_STATE_MAX_FRAME_SIZE];
    size_t frame_size = 0U;
    bool encoded = false;
    if (state == NULL || event == NULL || size != sizeof(*event) || event->reserved != 0U) {
        return -EINVAL;
    }
    state->native_event_receive_count += 1U;
    switch (event->event_type) {
        case RUNTIME_STATE_KERNEL_TOPOLOGY: {
            struct runtime_state_topology_event output;
            memcpy(&output, &event->payload.topology, sizeof(output));
            frame_size = RUNTIME_STATE_TOPOLOGY_FRAME_SIZE;
            encoded = runtime_state_encode_topology_frame(
                frame, sizeof(frame), &output, state->next_metadata_sequence);
            if (output.n_stream != 0U && output.n_layer != 0U
                && output.n_stream <= UINT64_MAX / (2U * output.n_layer)) {
                state->expected_tensor_count =
                    2U * output.n_stream * output.n_layer;
            }
            state->topology_event_count += 1U;
            break;
        }
        case RUNTIME_STATE_KERNEL_TENSOR_LAYOUT: {
            struct runtime_state_tensor_layout_event output;
            memcpy(&output, &event->payload.tensor, sizeof(output));
            frame_size = RUNTIME_STATE_TENSOR_FRAME_SIZE;
            encoded = runtime_state_encode_tensor_frame(
                frame, sizeof(frame), &output, state->next_metadata_sequence);
            state->tensor_layout_event_count += 1U;
            break;
        }
        case RUNTIME_STATE_KERNEL_KV_BACKING: {
            struct runtime_state_kv_backing_event output;
            memcpy(&output, &event->payload.backing, sizeof(output));
            frame_size = RUNTIME_STATE_BACKING_FRAME_SIZE;
            encoded = runtime_state_encode_backing_frame(
                frame, sizeof(frame), &output, state->next_metadata_sequence);
            state->kv_backing_event_count += 1U;
            break;
        }
        case RUNTIME_STATE_KERNEL_TOKEN_CELL: {
            struct runtime_state_token_cell_event output;
            memcpy(&output, &event->payload.token, sizeof(output));
            frame_size = RUNTIME_STATE_TOKEN_FRAME_SIZE;
            encoded = runtime_state_encode_token_frame(
                frame, sizeof(frame), &output, state->next_metadata_sequence);
            break;
        }
        case RUNTIME_STATE_KERNEL_TOKEN_CELL_RANGE: {
            struct runtime_state_token_cell_range_event output;
            memcpy(&output, &event->payload.token_range, sizeof(output));
            frame_size = RUNTIME_STATE_TOKEN_CELL_RANGE_FRAME_SIZE;
            encoded = runtime_state_encode_token_range_frame(
                frame, sizeof(frame), &output,
                state->next_metadata_sequence);
            break;
        }
        case RUNTIME_STATE_KERNEL_APPLY_COMPLETE: {
            struct runtime_state_apply_complete_event output;
            memcpy(&output, &event->payload.apply_complete, sizeof(output));
            frame_size = RUNTIME_STATE_APPLY_COMPLETE_FRAME_SIZE;
            encoded = runtime_state_encode_apply_complete_frame(
                frame, sizeof(frame), &output, state->next_metadata_sequence);
            break;
        }
        case RUNTIME_STATE_KERNEL_MUTATION: {
            struct runtime_state_mutation_event output;
            memcpy(&output, &event->payload.mutation, sizeof(output));
            frame_size = RUNTIME_STATE_MUTATION_FRAME_SIZE;
            encoded = runtime_state_encode_mutation_frame(
                frame, sizeof(frame), &output, state->next_metadata_sequence);
            state->mutation_event_count += 1U;
            break;
        }
        default:
            return -EINVAL;
    }
    if (!encoded || !write_frame(frame, frame_size)) {
        state->failed = true;
        return -EIO;
    }
    state->native_wire_frame_count += 1U;
    state->next_metadata_sequence += 1U;
    report_ready_if_complete(state);
    return 0;
}

static const char *gather_failure_stage_name(
    enum gather_failure_stage stage
) {
    switch (stage) {
    case GATHER_FAILURE_PLAN:
        return "plan";
    case GATHER_FAILURE_READY:
        return "ready";
    case GATHER_FAILURE_STOP:
        return "stop";
    case GATHER_FAILURE_MEMORY_READ:
        return "memory_read";
    case GATHER_FAILURE_FRAME_EMIT:
        return "frame_emit";
    case GATHER_FAILURE_SEAL_EMIT:
        return "seal_emit";
    case GATHER_FAILURE_RESUME:
        return "resume";
    case GATHER_FAILURE_NONE:
    default:
        return "none";
    }
}

static void record_gather_failure(
    struct collector_state *state,
    enum gather_failure_stage stage,
    int error_code
) {
    if (state == NULL || stage == GATHER_FAILURE_NONE
        || state->gather_failure_stage != GATHER_FAILURE_NONE) {
        return;
    }
    state->gather_failure_stage = stage;
    state->gather_failure_errno = error_code > 0 ? error_code : 0;
    state->gather_failure_pending_count = state->pending_capture_count;
}

static bool report_gather_failure(const struct collector_state *state) {
    if (state == NULL
        || state->gather_failure_stage == GATHER_FAILURE_NONE) {
        return true;
    }
    return fprintf(stderr,
            "GATHER_FAILED runtime_state stage=%s error=%d pending=%zu\n",
            gather_failure_stage_name(state->gather_failure_stage),
            state->gather_failure_errno,
            state->gather_failure_pending_count) >= 0
        && fflush(stderr) == 0;
}

static bool report_bulk_metrics(const struct collector_state *state) {
    if (state == NULL || state->target_stopped
        || state->target_stop_started_ns != 0U
        || state->target_boundary_started_ns != 0U
        || state->pending_capture_count != 0U
        || state->gathered_capture_count != 0U) {
        return false;
    }
    return fprintf(stderr,
            "BULK_METRICS runtime_state"
            " native_read_plan_count=%" PRIu64
            " native_read_syscall_count=%" PRIu64
            " native_copied_bytes=%" PRIu64
            " native_bulk_frame_count=%" PRIu64
            " native_bulk_payload_bytes=%" PRIu64
            " native_logical_fragment_count=%" PRIu64
            " native_logical_bytes=%" PRIu64
            " native_encode_write_ns=%" PRIu64
            " capture_boundary_hold_ns=%" PRIu64
            " capture_boundary_total_ns=%" PRIu64 "\n",
            state->native_read_plan_count,
            state->native_read_syscall_count,
            state->native_copied_bytes,
            state->native_bulk_frame_count,
            state->native_bulk_payload_bytes,
            state->native_logical_fragment_count,
            state->native_logical_bytes,
            state->native_encode_write_ns,
            state->capture_boundary_hold_ns,
            state->capture_boundary_total_ns) >= 0
        && fflush(stderr) == 0;
}

static struct pending_capture *find_pending_capture(
    struct collector_state *state,
    uint64_t apply_call_id
) {
    size_t index;
    if (state == NULL || apply_call_id == 0U) {
        return NULL;
    }
    for (index = 0U; index < RUNTIME_STATE_GATHER_MAX_PENDING; ++index) {
        if (state->pending_captures[index].apply_call_id == apply_call_id) {
            return &state->pending_captures[index];
        }
    }
    return NULL;
}

static struct pending_capture *allocate_pending_capture(
    struct collector_state *state,
    uint64_t apply_call_id
) {
    size_t index;
    if (state == NULL || apply_call_id == 0U
        || state->read_plan_capacity == 0U
        || state->pending_capture_count >= RUNTIME_STATE_GATHER_MAX_PENDING) {
        return NULL;
    }
    for (index = 0U; index < RUNTIME_STATE_GATHER_MAX_PENDING; ++index) {
        struct pending_capture *pending = &state->pending_captures[index];
        if (pending->apply_call_id != 0U) {
            continue;
        }
        if (pending->read_plans == NULL) {
            pending->read_plans = calloc(
                state->read_plan_capacity, sizeof(*pending->read_plans));
            if (pending->read_plans == NULL) {
                return NULL;
            }
        }
        pending->apply_call_id = apply_call_id;
        pending->read_plan_count = 0U;
        pending->planned_copy_byte_count = 0U;
        state->pending_capture_count += 1U;
        return pending;
    }
    return NULL;
}

static void release_pending_capture(
    struct collector_state *state,
    struct pending_capture *pending
) {
    if (state == NULL || pending == NULL || pending->apply_call_id == 0U) {
        return;
    }
    pending->apply_call_id = 0U;
    pending->read_plan_count = 0U;
    pending->planned_copy_byte_count = 0U;
    if (state->pending_capture_count > 0U) {
        state->pending_capture_count -= 1U;
    }
}

static bool queue_gathered_capture(
    struct collector_state *state,
    struct pending_capture *pending,
    const runtime_state_state_unit_apply_ready *ready,
    uint8_t *payload,
    size_t payload_size
) {
    struct gathered_capture *gathered;
    if (state == NULL || pending == NULL || ready == NULL || payload == NULL
        || pending->apply_call_id == 0U || pending->read_plans == NULL
        || pending->read_plan_count == 0U || payload_size == 0U
        || ready->apply_call_id != pending->apply_call_id
        || ready->copy_byte_count != payload_size
        || state->gathered_capture_count
            >= RUNTIME_STATE_GATHER_MAX_PENDING) {
        errno = EINVAL;
        return false;
    }
    gathered = &state->gathered_captures[state->gathered_capture_count];
    if (gathered->read_plans != NULL || gathered->payload != NULL) {
        errno = EBUSY;
        return false;
    }
    gathered->ready = *ready;
    gathered->read_plans = pending->read_plans;
    gathered->read_plan_count = pending->read_plan_count;
    gathered->payload = payload;
    gathered->payload_size = payload_size;
    pending->read_plans = NULL;
    state->gathered_capture_count += 1U;
    return true;
}

static void clear_gathered_captures(struct collector_state *state) {
    size_t index;
    if (state == NULL) {
        return;
    }
    for (index = 0U; index < state->gathered_capture_count; ++index) {
        struct gathered_capture *gathered = &state->gathered_captures[index];
        free(gathered->read_plans);
        free(gathered->payload);
        memset(gathered, 0, sizeof(*gathered));
    }
    state->gathered_capture_count = 0U;
}

static bool append_read_plan(
    struct collector_state *state,
    const struct runtime_state_kernel_state_unit_read_plan_event *event,
    size_t size
) {
    struct runtime_state_state_unit_read_plan plan;
    struct pending_capture *pending;
    if (state == NULL || event == NULL
        || size != sizeof(*event)
        || event->event_type != RUNTIME_STATE_KERNEL_STATE_UNIT_READ_PLAN
        || event->reserved != 0U) {
        record_gather_failure(state, GATHER_FAILURE_PLAN, EINVAL);
        return false;
    }
    memcpy(&plan, &event->apply_call_id, sizeof(plan));
    if (!runtime_state_valid_state_unit_read_plan(&plan)) {
        record_gather_failure(state, GATHER_FAILURE_PLAN, EINVAL);
        return false;
    }
    pending = find_pending_capture(state, plan.apply_call_id);
    if (pending == NULL) {
        pending = allocate_pending_capture(state, plan.apply_call_id);
    }
    if (pending == NULL
        || pending->read_plan_count >= state->read_plan_capacity
        || (pending->read_plan_count == 0U
            && plan.first_fragment_index != 0U)
        || (pending->read_plan_count != 0U
            && !runtime_state_state_unit_read_plan_follows(
                &pending->read_plans[pending->read_plan_count - 1U],
                &plan))
        || pending->planned_copy_byte_count
            > UINT64_MAX - plan.copy_byte_count) {
        record_gather_failure(state, GATHER_FAILURE_PLAN, errno);
        return false;
    }
    pending->read_plans[pending->read_plan_count] = plan;
    pending->read_plan_count += 1U;
    pending->planned_copy_byte_count += plan.copy_byte_count;
    if (!accumulate_u64(&state->native_read_plan_count, 1U)) {
        record_gather_failure(state, GATHER_FAILURE_PLAN, EOVERFLOW);
        return false;
    }
    return true;
}

static bool target_is_stopped(pid_t target_tgid) {
    char path[64];
    char line[256];
    FILE *status;
    bool stopped = false;
    if (snprintf(path, sizeof(path), "/proc/%ld/status", (long)target_tgid)
            <= 0) {
        return false;
    }
    status = fopen(path, "r");
    if (status == NULL) {
        return false;
    }
    while (fgets(line, sizeof(line), status) != NULL) {
        if (strncmp(line, "State:", 6U) == 0) {
            const char *cursor = line + 6U;
            while (*cursor == ' ' || *cursor == '\t') {
                cursor += 1;
            }
            stopped = *cursor == 'T' || *cursor == 't';
            break;
        }
    }
    (void)fclose(status);
    return stopped;
}

static bool wait_for_target_stop(struct collector_state *state) {
    const uint64_t started_ns = monotonic_ns();
    const struct timespec pause = {.tv_sec = 0, .tv_nsec = 1000000L};
    if (state == NULL || state->target_tgid <= 0 || started_ns == 0U) {
        return false;
    }
    while (monotonic_ns() - started_ns < RUNTIME_STATE_STOP_TIMEOUT_NS) {
        if (target_is_stopped(state->target_tgid)) {
            const uint64_t stopped_ns = monotonic_ns();
            if (stopped_ns == 0U || state->target_stop_started_ns != 0U) {
                return false;
            }
            state->target_stopped = true;
            state->target_stop_started_ns = stopped_ns;
            return true;
        }
        if (kill(state->target_tgid, 0) != 0) {
            return false;
        }
        (void)nanosleep(&pause, NULL);
    }
    return false;
}

static bool resume_stopped_target(struct collector_state *state) {
    uint64_t resumed_ns;
    if (state == NULL) {
        return false;
    }
    if (!state->target_stopped) {
        return true;
    }
    if (state->target_tgid <= 0
        || kill(state->target_tgid, SIGCONT) != 0) {
        return false;
    }
    resumed_ns = monotonic_ns();
    if (resumed_ns == 0U || state->target_stop_started_ns == 0U
        || state->target_boundary_started_ns == 0U
        || resumed_ns < state->target_stop_started_ns
        || resumed_ns < state->target_boundary_started_ns
        || !accumulate_u64(
            &state->capture_boundary_hold_ns,
            resumed_ns - state->target_stop_started_ns)
        || !accumulate_u64(
            &state->capture_boundary_total_ns,
            resumed_ns - state->target_boundary_started_ns)) {
        return false;
    }
    state->target_stopped = false;
    state->target_stop_started_ns = 0U;
    state->target_boundary_started_ns = 0U;
    return true;
}

static bool gather_read_plan_into(
    struct collector_state *state,
    const struct runtime_state_state_unit_read_plan *plan,
    uint8_t *payload,
    size_t payload_size
) {
    struct iovec remote_iov[RUNTIME_STATE_GATHER_IOV_MAX];
    size_t span_count;
    size_t span_index = 0U;
    size_t payload_offset = 0U;
    if (state == NULL || plan == NULL || payload == NULL
        || !runtime_state_valid_state_unit_read_plan(plan)
        || plan->copy_byte_count > SIZE_MAX
        || payload_size != (size_t)plan->copy_byte_count) {
        errno = EINVAL;
        return false;
    }
    span_count = plan->transposed_value != 0U
        ? (size_t)plan->embedding_width : 1U;
    while (span_index < span_count) {
        struct iovec local_iov;
        size_t remote_iov_count = 0U;
        size_t batch_bytes = 0U;
        while (span_index + remote_iov_count < span_count
                && remote_iov_count < RUNTIME_STATE_GATHER_IOV_MAX) {
            const size_t dimension = span_index + remote_iov_count;
            uint64_t remote_offset;
            uint64_t remote_address;
            size_t span_bytes;
            if (plan->transposed_value != 0U) {
                const uint64_t dimension_stride
                    = (uint64_t)plan->cell_capacity * plan->element_size;
                remote_offset = (uint64_t)dimension * dimension_stride
                    + plan->first_cell_index * plan->element_size;
                span_bytes = (size_t)plan->token_count
                    * plan->element_size;
            } else {
                remote_offset = plan->first_cell_index * plan->row_size;
                span_bytes = (size_t)plan->copy_byte_count;
            }
            if (remote_offset > UINT64_MAX - plan->tensor_address
                || batch_bytes > SIZE_MAX - span_bytes) {
                errno = EOVERFLOW;
                return false;
            }
            remote_address = plan->tensor_address + remote_offset;
            remote_iov[remote_iov_count].iov_base
                = (void *)(uintptr_t)remote_address;
            remote_iov[remote_iov_count].iov_len = span_bytes;
            batch_bytes += span_bytes;
            remote_iov_count += 1U;
        }
        if (remote_iov_count == 0U
            || batch_bytes > (size_t)plan->copy_byte_count
            || payload_offset
                > (size_t)plan->copy_byte_count - batch_bytes) {
            errno = EOVERFLOW;
            return false;
        }
        local_iov.iov_base = payload + payload_offset;
        local_iov.iov_len = batch_bytes;
        {
            const ssize_t read_bytes = process_vm_readv(
                state->target_tgid,
                &local_iov,
                1U,
                remote_iov,
                remote_iov_count,
                0U);
            if (read_bytes != (ssize_t)batch_bytes) {
                if (read_bytes >= 0) {
                    errno = EIO;
                }
                return false;
            }
            if (!accumulate_u64(&state->native_read_syscall_count, 1U)
                || !accumulate_u64(
                    &state->native_copied_bytes, (uint64_t)batch_bytes)) {
                errno = EOVERFLOW;
                return false;
            }
        }
        payload_offset += batch_bytes;
        span_index += remote_iov_count;
    }
    if (payload_offset != (size_t)plan->copy_byte_count) {
        errno = EIO;
        return false;
    }
    return true;
}

static bool emit_gathered_bulk_frames(
    struct collector_state *state,
    const struct runtime_state_state_unit_read_plan *plan,
    const uint8_t *payload
) {
    uint8_t frame[RUNTIME_STATE_MAX_FRAME_SIZE];
    uint32_t first_fragment_offset = 0U;
    if (state == NULL || plan == NULL || payload == NULL
        || !runtime_state_valid_state_unit_read_plan(plan)) {
        return false;
    }
    while (first_fragment_offset < plan->fragment_count) {
        struct runtime_state_bulk_window window;
        struct runtime_state_state_unit_bulk bulk;
        size_t frame_size = 0U;
        uint64_t started_ns;
        uint64_t completed_ns;
        started_ns = monotonic_ns();
        if (!runtime_state_bulk_window_for_plan(
                plan, first_fragment_offset,
                RUNTIME_STATE_BULK_PAYLOAD_BYTES, &window)
            || window.payload_offset > plan->copy_byte_count
            || window.payload_size
                > plan->copy_byte_count - window.payload_offset
            || !runtime_state_bulk_from_plan(plan, &window, &bulk)
            || !runtime_state_encode_state_unit_bulk_frame(
                frame, sizeof(frame), &bulk,
                payload + (size_t)window.payload_offset,
                state->next_state_unit_sequence, &frame_size)
            || !write_frame(frame, frame_size)) {
            return false;
        }
        completed_ns = monotonic_ns();
        if (started_ns == 0U || completed_ns == 0U
            || completed_ns < started_ns
            || !accumulate_u64(&state->native_bulk_frame_count, 1U)
            || !accumulate_u64(
                &state->native_bulk_payload_bytes,
                (uint64_t)window.payload_size)
            || !accumulate_u64(
                &state->native_logical_fragment_count,
                (uint64_t)window.fragment_count)
            || !accumulate_u64(
                &state->native_logical_bytes,
                (uint64_t)window.payload_size)
            || !accumulate_u64(
                &state->native_encode_write_ns,
                completed_ns - started_ns)) {
            errno = EOVERFLOW;
            return false;
        }
        state->native_wire_frame_count += 1U;
        state->next_state_unit_sequence += 1U;
        first_fragment_offset
            = window.first_fragment_offset + window.fragment_count;
    }
    return first_fragment_offset == plan->fragment_count;
}

static bool flush_gathered_captures(struct collector_state *state) {
    size_t capture_index;
    bool success = true;
    if (state == NULL || state->target_stopped
        || state->pending_capture_count != 0U) {
        errno = EBUSY;
        return false;
    }
    for (capture_index = 0U;
            capture_index < state->gathered_capture_count;
            ++capture_index) {
        struct gathered_capture *gathered
            = &state->gathered_captures[capture_index];
        uint8_t seal_frame[RUNTIME_STATE_STATE_UNIT_SEAL_FRAME_SIZE];
        size_t payload_offset = 0U;
        size_t plan_index;
        if (gathered->read_plans == NULL || gathered->read_plan_count == 0U
            || gathered->payload == NULL || gathered->payload_size == 0U
            || gathered->read_plan_count > UINT32_MAX
            || !runtime_state_valid_state_unit_apply_ready(
                &gathered->ready,
                &gathered->read_plans[gathered->read_plan_count - 1U],
                (uint32_t)gathered->read_plan_count,
                gathered->payload_size)) {
            record_gather_failure(state, GATHER_FAILURE_READY, EINVAL);
            success = false;
            break;
        }
        for (plan_index = 0U;
                plan_index < gathered->read_plan_count;
                ++plan_index) {
            const struct runtime_state_state_unit_read_plan *plan
                = &gathered->read_plans[plan_index];
            if (plan->copy_byte_count > SIZE_MAX
                || payload_offset > gathered->payload_size
                || (size_t)plan->copy_byte_count
                    > gathered->payload_size - payload_offset
                || !emit_gathered_bulk_frames(
                    state, plan, gathered->payload + payload_offset)) {
                record_gather_failure(
                    state, GATHER_FAILURE_FRAME_EMIT, errno);
                success = false;
                break;
            }
            payload_offset += (size_t)plan->copy_byte_count;
        }
        if (!success) {
            break;
        }
        if (payload_offset != gathered->payload_size
            || !runtime_state_encode_state_unit_seal_frame(
                seal_frame, sizeof(seal_frame), &gathered->ready,
                state->next_state_unit_sequence)
            || !write_frame(seal_frame, sizeof(seal_frame))) {
            record_gather_failure(state, GATHER_FAILURE_SEAL_EMIT, errno);
            success = false;
            break;
        }
        state->native_wire_frame_count += 1U;
        state->next_state_unit_sequence += 1U;
    }
    clear_gathered_captures(state);
    return success;
}

static bool capture_ready_apply(
    struct collector_state *state,
    const struct runtime_state_kernel_state_unit_apply_ready_event *event,
    size_t size
) {
    runtime_state_state_unit_apply_ready ready;
    struct pending_capture *pending = NULL;
    uint8_t *payload = NULL;
    size_t payload_offset = 0U;
    size_t plan_index;
    bool success = false;
    if (state == NULL || event == NULL || size != sizeof(*event)
        || event->event_type != RUNTIME_STATE_KERNEL_STATE_UNIT_APPLY_READY
        || event->reserved != 0U) {
        record_gather_failure(state, GATHER_FAILURE_READY, EINVAL);
        goto cleanup_capture;
    }
    memcpy(&ready, &event->payload, sizeof(ready));
    pending = find_pending_capture(state, ready.apply_call_id);
    if (pending == NULL || pending->read_plan_count == 0U
        || !runtime_state_valid_state_unit_apply_ready(
            &ready,
            &pending->read_plans[pending->read_plan_count - 1U],
            (uint32_t)pending->read_plan_count,
            pending->planned_copy_byte_count)) {
        record_gather_failure(state, GATHER_FAILURE_READY, EINVAL);
        goto cleanup_capture;
    }
    if (state->target_boundary_started_ns != 0U) {
        record_gather_failure(state, GATHER_FAILURE_READY, EBUSY);
        goto cleanup_capture;
    }
    state->target_boundary_started_ns = ready.writer_return_monotonic_ns;
    if (!wait_for_target_stop(state)) {
        record_gather_failure(state, GATHER_FAILURE_STOP, errno);
        goto cleanup_capture;
    }
    if (pending->planned_copy_byte_count > SIZE_MAX) {
        record_gather_failure(state, GATHER_FAILURE_MEMORY_READ, EOVERFLOW);
        goto cleanup_capture;
    }
    payload = malloc((size_t)pending->planned_copy_byte_count);
    if (payload == NULL) {
        record_gather_failure(state, GATHER_FAILURE_MEMORY_READ, ENOMEM);
        goto cleanup_capture;
    }
    for (plan_index = 0U;
            plan_index < pending->read_plan_count;
            ++plan_index) {
        const struct runtime_state_state_unit_read_plan *plan
            = &pending->read_plans[plan_index];
        if (plan->copy_byte_count > SIZE_MAX
            || payload_offset > (size_t)pending->planned_copy_byte_count
            || (size_t)plan->copy_byte_count
                > (size_t)pending->planned_copy_byte_count - payload_offset
            || !gather_read_plan_into(
                state, plan, payload + payload_offset,
                (size_t)plan->copy_byte_count)) {
            record_gather_failure(
                state, GATHER_FAILURE_MEMORY_READ, errno);
            goto cleanup_capture;
        }
        payload_offset += (size_t)plan->copy_byte_count;
    }
    if (payload_offset != (size_t)pending->planned_copy_byte_count) {
        record_gather_failure(state, GATHER_FAILURE_MEMORY_READ, EIO);
        goto cleanup_capture;
    }
    if (!queue_gathered_capture(
            state, pending, &ready, payload, payload_offset)) {
        record_gather_failure(state, GATHER_FAILURE_MEMORY_READ, errno);
        goto cleanup_capture;
    }
    payload = NULL;
    success = true;

cleanup_capture:
    free(payload);
    if (state != NULL) {
        release_pending_capture(state, pending);
    }
    return success;
}

static int collect_state_unit_event(void *context, void *data, size_t size) {
    struct collector_state *state = context;
    const struct runtime_state_kernel_state_unit_event_header *header = data;
    uint8_t frame[RUNTIME_STATE_MAX_FRAME_SIZE];
    size_t frame_size = 0U;
    bool encoded = false;
    if (state == NULL || header == NULL || size < sizeof(*header)
        || header->reserved != 0U) {
        return -EINVAL;
    }
    if (header->event_type == RUNTIME_STATE_KERNEL_STATE_UNIT_READ_PLAN) {
        const struct runtime_state_kernel_state_unit_read_plan_event *event
            = data;
        state->native_event_receive_count += 1U;
        if (!append_read_plan(state, event, size)) {
            state->failed = true;
            return -EINVAL;
        }
        return 0;
    } else if (header->event_type
            == RUNTIME_STATE_KERNEL_STATE_UNIT_APPLY_READY) {
        const struct runtime_state_kernel_state_unit_apply_ready_event *event
            = data;
        state->native_event_receive_count += 1U;
        if (!capture_ready_apply(state, event, size)) {
            state->failed = true;
            return -EIO;
        }
        return 0;
    } else if (header->event_type
            == RUNTIME_STATE_KERNEL_STATE_UNIT_SCHEDULE_BATCH) {
        const struct runtime_state_kernel_state_unit_schedule_batch_event *event
            = data;
        if (state->active_stream_count == 0U
            || state->active_stream_count > 8U
            || event->stream_index >= state->active_stream_count) {
            return -EINVAL;
        }
        encoded = runtime_state_encode_kernel_schedule_batch_frame(
            frame, sizeof(frame), event, size,
            state->next_state_unit_sequence, &frame_size);
    } else if (header->event_type
            == RUNTIME_STATE_KERNEL_STATE_UNIT_FRAGMENT_BATCH) {
        const struct runtime_state_kernel_state_unit_fragment_batch_event *event
            = data;
        uint32_t index;
        if (state->active_stream_count == 0U
            || state->active_stream_count > 8U
            || event->record_count == 0U
            || event->record_count > RUNTIME_STATE_MAX_BATCH_FRAGMENTS) {
            return -EINVAL;
        }
        for (index = 0U; index < event->record_count; ++index) {
            if (event->fragments[index].stream_index
                    >= state->active_stream_count) {
                return -EINVAL;
            }
        }
        encoded = runtime_state_encode_kernel_fragment_batch_frame(
            frame, sizeof(frame), event, size,
            state->next_state_unit_sequence, &frame_size);
    } else if (header->event_type
            == RUNTIME_STATE_KERNEL_STATE_UNIT_FRAGMENT) {
        const struct runtime_state_kernel_state_unit_fragment_event *event = data;
        struct runtime_state_state_unit_fragment fragment;
        const struct runtime_state_kernel_state_unit_fragment *source =
            &event->payload;
        if (size < RUNTIME_STATE_KERNEL_STATE_UNIT_FRAGMENT_EVENT_PREFIX_SIZE) {
            return -EINVAL;
        }
        if (source->payload_size == 0U
            || source->payload_size > RUNTIME_STATE_MAX_FRAGMENT_BYTES
            || source->stream_index >= state->active_stream_count) {
            return -EINVAL;
        }
        if (size != RUNTIME_STATE_KERNEL_STATE_UNIT_FRAGMENT_EVENT_PREFIX_SIZE
                + (size_t)source->payload_size) {
            return -EINVAL;
        }
        memcpy(&fragment, source, sizeof(fragment));
        frame_size = RUNTIME_STATE_HEADER_SIZE
            + RUNTIME_STATE_STATE_UNIT_FRAGMENT_METADATA_SIZE
            + source->payload_size;
        encoded = runtime_state_encode_state_unit_fragment_frame(
            frame, sizeof(frame), &fragment, source->payload,
            state->next_state_unit_sequence);
    } else if (header->event_type == RUNTIME_STATE_KERNEL_STATE_UNIT_SEAL) {
        const struct runtime_state_kernel_state_unit_seal_event *event = data;
        struct runtime_state_state_unit_seal seal;
        if (size != sizeof(struct runtime_state_kernel_state_unit_seal_event)) {
            return -EINVAL;
        }
        memcpy(&seal, &event->payload, sizeof(seal));
        if (state->active_stream_count == 0U
            || state->active_stream_count > 8U
            || (seal.stream_mask
                & ~((UINT32_C(1) << state->active_stream_count) - 1U))
                != 0U) {
            return -EINVAL;
        }
        frame_size = RUNTIME_STATE_STATE_UNIT_SEAL_FRAME_SIZE;
        encoded = runtime_state_encode_state_unit_seal_frame(
            frame, sizeof(frame), &seal,
            state->next_state_unit_sequence);
    } else {
        return -EINVAL;
    }
    state->native_event_receive_count += 1U;
    if (!encoded || !write_frame(frame, frame_size)) {
        state->failed = true;
        return -EIO;
    }
    state->native_wire_frame_count += 1U;
    state->next_state_unit_sequence += 1U;
    return 0;
}

static struct bpf_link *attach_one(
    struct bpf_object *object,
    const char *program_name,
    const char *binary,
    uint64_t offset,
    pid_t target,
    bool return_probe,
    uint64_t cookie
) {
    struct bpf_program *program = bpf_object__find_program_by_name(object, program_name);
    struct bpf_uprobe_opts options = {
        .sz = sizeof(options),
        .retprobe = return_probe,
        .bpf_cookie = cookie,
    };
    if (program == NULL) {
        return NULL;
    }
    return bpf_program__attach_uprobe_opts(program, target, binary, (size_t)offset, &options);
}

static int map_fd_by_name(struct bpf_object *object, const char *name) {
    struct bpf_map *map = bpf_object__find_map_by_name(object, name);
    return map == NULL ? -ENOENT : bpf_map__fd(map);
}

static bool configure_map_capacities(
    struct bpf_object *object,
    uint32_t expected_n_stream,
    uint32_t expected_n_layer
) {
    struct bpf_map *tensor_candidates;
    struct bpf_map *retained_tensor_layouts;
    struct bpf_map *stream_versions;
    struct bpf_map *observed_sequences;
    struct bpf_map *payload_events;
    uint32_t tensor_entry_count;
    uint32_t payload_ring_byte_count;
    if (object == NULL
        || (expected_n_stream != 4U && expected_n_stream != 8U)) {
        return false;
    }
    if (expected_n_layer == 0U) {
        tensor_entry_count = RUNTIME_STATE_MAX_TENSOR_EVENTS;
    } else {
        if (expected_n_layer > RUNTIME_STATE_MAX_TENSOR_EVENTS
                / (2U * expected_n_stream)) {
            return false;
        }
        tensor_entry_count = 2U * expected_n_stream * expected_n_layer;
    }
    payload_ring_byte_count = expected_n_stream
        * RUNTIME_STATE_PAYLOAD_RING_BYTES_PER_STREAM;
    tensor_candidates = bpf_object__find_map_by_name(
        object, "tensor_candidates");
    retained_tensor_layouts = bpf_object__find_map_by_name(
        object, "runtime_state_retained_tensor_layouts");
    stream_versions = bpf_object__find_map_by_name(
        object, "runtime_state_stream_versions");
    observed_sequences = bpf_object__find_map_by_name(
        object, "observed_sequence_stream");
    payload_events = bpf_object__find_map_by_name(
        object, "runtime_state_payload_events");
    if (tensor_candidates == NULL || retained_tensor_layouts == NULL
        || stream_versions == NULL || observed_sequences == NULL
        || payload_events == NULL
        || bpf_map__set_max_entries(
            tensor_candidates, tensor_entry_count) != 0
        || bpf_map__set_max_entries(
            retained_tensor_layouts, tensor_entry_count) != 0
        || bpf_map__set_max_entries(
            stream_versions, expected_n_stream) != 0
        || bpf_map__set_max_entries(
            observed_sequences, expected_n_stream) != 0
        || bpf_map__set_max_entries(
            payload_events, payload_ring_byte_count) != 0) {
        return false;
    }
    return true;
}

static uint64_t monotonic_ns(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0U;
    }
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
}

static bool drain_to_counter_parity(
    struct ring_buffer *ring,
    int counter_fd,
    struct collector_state *collector,
    struct runtime_state_bpf_counters *counters
) {
    __u32 zero = 0U;
    const uint64_t started = monotonic_ns();
    const uint64_t deadline = started + RUNTIME_STATE_DRAIN_TIMEOUT_NS;
    bool stall_reported = false;
    if (started == 0U || deadline < started) {
        return false;
    }
    for (;;) {
        int drained = ring_buffer__poll(ring, 1);
        if (drained < 0 && drained != -EINTR) {
            return false;
        }
        if (bpf_map_lookup_elem(counter_fd, &zero, counters) != 0) {
            return false;
        }
        if (counters->kernel_event_submit_count
                + counters->state_unit_fragment_submit_count
                + counters->state_unit_seal_submit_count
            == collector->native_event_receive_count
            && counters->active_emitter_count == 0U
            && counters->inflight_call_count == 0U
            && collector->pending_capture_count == 0U) {
            return true;
        }
        {
            const uint64_t now = monotonic_ns();
            if (!stall_reported
                && now - started >= RUNTIME_STATE_DRAIN_STALL_DIAGNOSTIC_NS) {
                const uint64_t submitted
                    = counters->kernel_event_submit_count
                        + counters->state_unit_fragment_submit_count
                        + counters->state_unit_seal_submit_count;
                if (fprintf(stderr,
                        "DRAIN_STALLED runtime_state submitted=%" PRIu64
                        " received=%" PRIu64 " active=%" PRIu64
                        " inflight=%" PRIu64 " pending=%zu stopped=%u"
                        " plans=%" PRIu64 " ready=%" PRIu64
                        " reserve=%" PRIu64 " unmatched=%" PRIu64 "\n",
                        submitted,
                        (uint64_t)collector->native_event_receive_count,
                        (uint64_t)counters->active_emitter_count,
                        (uint64_t)counters->inflight_call_count,
                        collector->pending_capture_count,
                        collector->target_stopped ? 1U : 0U,
                        (uint64_t)counters->state_unit_fragment_submit_count,
                        (uint64_t)counters->state_unit_seal_submit_count,
                        (uint64_t)counters->payload_ringbuf_reserve_fail_count,
                        (uint64_t)counters->unmatched_call_count) < 0
                    || fflush(stderr) != 0) {
                    return false;
                }
                stall_reported = true;
            }
            if (now >= deadline) {
                return false;
            }
        }
    }
}

static bool accumulate_u64(uint64_t *total, uint64_t value) {
    if (total == NULL || *total > UINT64_MAX - value) {
        return false;
    }
    *total += value;
    return true;
}

static bool capture_stream_version_snapshot(
    struct ring_buffer *ring,
    int counter_fd,
    int stream_version_fd,
    struct collector_state *collector,
    struct runtime_state_bpf_counters *counters,
    struct runtime_state_stream_version_snapshot *snapshot
) {
    size_t attempt;
    if (ring == NULL || counter_fd < 0 || stream_version_fd < 0
        || collector == NULL || counters == NULL || snapshot == NULL) {
        return false;
    }
    for (attempt = 0U; attempt < 16U; ++attempt) {
        struct runtime_state_bpf_counters after = {0};
        __u32 zero = 0U;
        size_t stream_index;
        if (!drain_to_counter_parity(ring, counter_fd, collector, counters)) {
            return false;
        }
        memset(snapshot, 0, sizeof(*snapshot));
        snapshot->active_stream_count = collector->active_stream_count;
        for (stream_index = 0U;
             stream_index < collector->active_stream_count;
             ++stream_index) {
            __u32 key = (__u32)stream_index;
            __u64 encoded = 0U;
            if (bpf_map_lookup_elem(stream_version_fd, &key, &encoded) != 0) {
                return false;
            }
            if ((encoded & RUNTIME_STATE_STREAM_VERSION_BUSY_FLAG) != 0U) {
                snapshot->stream_states[stream_index] =
                    RUNTIME_STATE_STREAM_BUSY;
                snapshot->stream_apply_call_ids[stream_index] =
                    encoded & ~RUNTIME_STATE_STREAM_VERSION_BUSY_FLAG;
            } else if (encoded != 0U) {
                snapshot->stream_states[stream_index] =
                    RUNTIME_STATE_STREAM_READY;
                snapshot->stream_apply_call_ids[stream_index] = encoded;
            } else {
                snapshot->stream_states[stream_index] =
                    RUNTIME_STATE_STREAM_EMPTY;
            }
        }
        if (bpf_map_lookup_elem(counter_fd, &zero, &after) != 0) {
            return false;
        }
        if (after.kernel_event_submit_count != counters->kernel_event_submit_count
            || after.state_unit_fragment_submit_count
                != counters->state_unit_fragment_submit_count
            || after.state_unit_seal_submit_count
                != counters->state_unit_seal_submit_count
            || after.topology_event_count != counters->topology_event_count
            || after.tensor_layout_event_count != counters->tensor_layout_event_count
            || after.kv_backing_event_count != counters->kv_backing_event_count
            || after.mutation_event_count != counters->mutation_event_count
            || after.kernel_event_submit_count
                    + after.state_unit_fragment_submit_count
                    + after.state_unit_seal_submit_count
                != collector->native_event_receive_count
            || after.active_emitter_count != 0U
            || after.inflight_call_count != 0U) {
            continue;
        }
        *counters = after;
        collector->next_snapshot_id += 1U;
        if (collector->next_snapshot_id == 0U) {
            return false;
        }
        snapshot->snapshot_id = collector->next_snapshot_id;
        snapshot->mutation_generation = after.mutation_event_count;
        snapshot->backing_generation = after.kv_backing_event_count;
        snapshot->layout_generation = after.tensor_layout_event_count;
        return runtime_state_valid_stream_version(snapshot);
    }
    return false;
}

static void copy_counters(
    const struct runtime_state_bpf_counters *source,
    const struct collector_state *collector,
    struct runtime_state_terminal_counters *destination,
    uint64_t cutoff_monotonic_ns
) {
    memset(destination, 0, sizeof(*destination));
    destination->cutoff_monotonic_ns = cutoff_monotonic_ns;
    destination->topology_event_count = source->topology_event_count;
    destination->tensor_layout_event_count = source->tensor_layout_event_count;
    destination->kv_backing_event_count = source->kv_backing_event_count;
    destination->token_cell_event_count = source->token_cell_event_count;
    destination->apply_complete_event_count = source->apply_complete_event_count;
    destination->kernel_event_submit_count = source->kernel_event_submit_count
        + source->state_unit_fragment_submit_count
        + source->state_unit_seal_submit_count;
    destination->native_event_receive_count = collector->native_event_receive_count;
    destination->native_wire_frame_count = collector->native_wire_frame_count;
    destination->ringbuf_reserve_fail_count = source->ringbuf_reserve_fail_count
        + source->payload_ringbuf_reserve_fail_count;
    destination->user_memory_read_failure_count = source->user_memory_read_failure_count
        + source->payload_user_memory_read_failure_count;
    destination->unmatched_call_count = source->unmatched_call_count;
    destination->unsupported_mutation_count = source->unsupported_mutation_count;
    destination->mapping_mutation_count = source->mapping_mutation_count;
    destination->final_inflight_count = source->inflight_call_count;
    destination->mutation_event_count = source->mutation_event_count;
    destination->active_emitter_count = source->active_emitter_count;
}

static bool write_counter_frame(
    struct collector_state *collector,
    const struct runtime_state_terminal_counters *counters,
    bool terminal
) {
    uint8_t frame[RUNTIME_STATE_COUNTERS_FRAME_SIZE];
    const bool encoded = terminal
        ? runtime_state_encode_terminal_frame(
            frame, sizeof(frame), counters, collector->next_metadata_sequence)
        : runtime_state_encode_watermark_frame(
            frame, sizeof(frame), counters, collector->next_metadata_sequence);
    if (!encoded || !write_frame(frame, sizeof(frame)) || fflush(stdout) != 0) {
        return false;
    }
    collector->next_metadata_sequence += 1U;
    return true;
}

static bool write_stream_version_frame(
    struct collector_state *collector,
    const struct runtime_state_stream_version_snapshot *snapshot
) {
    uint8_t frame[RUNTIME_STATE_STREAM_VERSION_FRAME_SIZE];
    if (collector == NULL || snapshot == NULL
        || !runtime_state_encode_stream_version_frame(
            frame, sizeof(frame), snapshot, collector->next_metadata_sequence)
        || !write_frame(frame, sizeof(frame))) {
        return false;
    }
    collector->next_metadata_sequence += 1U;
    return true;
}

int main(int argc, char **argv) {
    struct observer_options options;
    struct runtime_state_target_config target = {0};
    struct runtime_state_bpf_counters counters = {0};
    struct runtime_state_terminal_counters terminal = {0};
    struct runtime_state_stream_version_snapshot stream_version = {0};
    struct collector_state collector = {
        .next_metadata_sequence = 1U,
        .next_state_unit_sequence = 1U,
        .next_snapshot_id = 0U,
    };
    struct bpf_object *object = NULL;
    struct ring_buffer *ring = NULL;
    struct bpf_link *links[MAX_LINK_COUNT] = {0};
    const struct observer_base_link base_links[WRITER_BASE_LINK_COUNT] = {
        {"observe_kv_constructor_entry", 0U, false},
        {"observe_kv_constructor_return", 0U, true},
        {"observe_n_pos_per_embd_entry", 1U, false},
        {"observe_n_pos_per_embd_return", 1U, true},
        {"observe_view_2d_entry", 2U, false},
        {"observe_view_2d_return", 2U, true},
        {"observe_buffer_base_entry", 3U, false},
        {"observe_buffer_base_return", 3U, true},
        {"observe_buffer_size_entry", 4U, false},
        {"observe_buffer_size_return", 4U, true},
        {"observe_apply_topology_bootstrap", 5U, false},
        {"observe_apply_ubatch_entry", 5U, false},
        {"observe_apply_ubatch_return", 5U, true},
        {"observe_process_ubatch_entry", 6U, false},
        {"observe_process_ubatch_return", 6U, true},
    };
    size_t base_link_count;
    size_t link_count = 0U;
    int map_fd;
    int counter_fd;
    int diagnostic_fd;
    int event_fd;
    int payload_event_fd;
    int stream_version_fd;
    __u32 zero = 0U;
    size_t index;
    bool mutations_armed = false;
    int result = 1;
    struct sigaction action = {0};
    if (!parse_options(argc, argv, &options)) {
        usage(argv[0]);
        return 2;
    }
    collector.target_tgid = (pid_t)options.target_tgid;
    collector.read_plan_capacity = RUNTIME_STATE_GATHER_MAX_PLANS;
    if ((options.capture_mode == OBSERVER_CAPTURE_WRITER_BOUNDARY
            || options.capture_mode
                == OBSERVER_CAPTURE_GATHER_WRITER_BOUNDARY)
        && setvbuf(
            stdout,
            writer_stdout_buffer,
            _IOFBF,
            sizeof(writer_stdout_buffer)) != 0) {
        fprintf(stderr, "runtime-state observer failed to configure writer output buffer\n");
        return 2;
    }
    collector.active_stream_count = options.expected_n_stream;
    base_link_count = options.capture_mode == OBSERVER_CAPTURE_WRITER_BOUNDARY
            || options.capture_mode
                == OBSERVER_CAPTURE_GATHER_WRITER_BOUNDARY
        ? WRITER_BASE_LINK_COUNT
        : METADATA_BASE_LINK_COUNT;
    link_count = base_link_count;
    target.target_tgid = options.target_tgid;
    target.expected_n_stream = options.expected_n_stream;
    target.expected_n_seq_max = options.expected_n_seq_max;
    target.expected_n_layer = options.expected_n_layer;
    target.tensor_data_offset = options.tensor_data_offset;
    target.tensor_type_offset = options.tensor_type_offset;
    target.capture_mode = (uint32_t)options.capture_mode;
    target.capture_armed = 0U;

    action.sa_handler = handle_signal;
    sigemptyset(&action.sa_mask);
    (void)sigaction(SIGINT, &action, NULL);
    (void)sigaction(SIGTERM, &action, NULL);
    libbpf_set_strict_mode(LIBBPF_STRICT_ALL);
    object = bpf_object__open_file(options.bpf_object, NULL);
    if (object == NULL || libbpf_get_error(object) != 0) {
        object = NULL;
        fprintf(stderr, "runtime-state observer failed to open BPF object\n");
        goto cleanup;
    }
    if (!configure_map_capacities(
            object, options.expected_n_stream, options.expected_n_layer)) {
        fprintf(stderr, "runtime-state observer failed to size BPF maps\n");
        goto cleanup;
    }
    if (bpf_object__load(object) != 0) {
        fprintf(stderr, "runtime-state observer failed to load BPF object\n");
        goto cleanup;
    }
    map_fd = map_fd_by_name(object, "runtime_state_target");
    counter_fd = map_fd_by_name(object, "runtime_state_counters");
    diagnostic_fd = map_fd_by_name(object, "runtime_state_diagnostics");
    event_fd = map_fd_by_name(object, "runtime_state_events");
    payload_event_fd = map_fd_by_name(
        object, "runtime_state_payload_events");
    stream_version_fd = map_fd_by_name(
        object, "runtime_state_stream_versions");
    if (map_fd < 0 || counter_fd < 0 || diagnostic_fd < 0 || event_fd < 0
        || payload_event_fd < 0
        || stream_version_fd < 0
        || bpf_map_update_elem(map_fd, &zero, &target, BPF_ANY) != 0) {
        fprintf(stderr, "runtime-state observer failed to configure maps\n");
        goto cleanup;
    }
    for (index = 0U; index < base_link_count; ++index) {
        links[index] = attach_one(
            object,
            base_links[index].program_name,
            options.probes[base_links[index].probe_index].binary,
            options.probes[base_links[index].probe_index].offset,
            (pid_t)options.target_tgid,
            base_links[index].return_probe,
            0U);
        if (links[index] == NULL || libbpf_get_error(links[index]) != 0) {
            links[index] = NULL;
            fprintf(stderr, "runtime-state observer failed to attach qualified probes\n");
            goto cleanup;
        }
    }
    ring = ring_buffer__new(event_fd, collect_event, &collector, NULL);
    if (ring == NULL
        || ring_buffer__add(
            ring, payload_event_fd,
            collect_state_unit_event, &collector) != 0) {
        fprintf(stderr, "runtime-state observer failed to create ring consumer\n");
        goto cleanup;
    }
    for (index = 0U; index < options.probe_identity_count; ++index) {
        if (fprintf(stderr, "ATTACHED_PROBE runtime_state %s\n",
                options.probe_identities[index]) < 0) {
            goto cleanup;
        }
    }
    if (fprintf(stderr, "ATTACHED runtime_state probes=%zu identities=%zu\n",
            link_count, options.probe_identity_count) < 0
        || fflush(stderr) != 0) {
        goto cleanup;
    }
    while (!stop_requested && !collector.failed) {
        struct pollfd input = {.fd = STDIN_FILENO, .events = POLLIN | POLLHUP};
        char command[32];
        int poll_result = ring_buffer__poll(ring, 100);
        if (poll_result < 0 && poll_result != -EINTR) {
            collector.failed = true;
            break;
        }
        if (collector.target_stopped
            && collector.pending_capture_count == 0U) {
            if (!resume_stopped_target(&collector)) {
                record_gather_failure(
                    &collector, GATHER_FAILURE_RESUME, errno);
                collector.failed = true;
                break;
            }
            if (!flush_gathered_captures(&collector)) {
                collector.failed = true;
                break;
            }
        }
        if (poll(&input, 1U, 0) > 0) {
            if ((input.revents & POLLIN) != 0) {
                if (fgets(command, sizeof(command), stdin) == NULL
                    || strcmp(command, "STOP\n") == 0) {
                    break;
                }
                if (strcmp(command, "ARM\n") == 0) {
                    if (mutations_armed || !collector.ready_reported) {
                        collector.failed = true;
                        break;
                    }
                    for (index = 0U; index < options.mutation_count; ++index) {
                        const size_t link_index = base_link_count + index;
                        links[link_index] = attach_one(
                            object,
                            "observe_forbidden_mutation",
                            options.mutations[index].binary,
                            options.mutations[index].offset,
                            (pid_t)options.target_tgid,
                            false,
                            options.mutations[index].mutation_kind);
                        if (links[link_index] == NULL
                            || libbpf_get_error(links[link_index]) != 0) {
                            links[link_index] = NULL;
                            collector.failed = true;
                            break;
                        }
                    }
                    if (collector.failed) {
                        fprintf(stderr,
                            "runtime-state observer failed to attach mutation probes\n");
                        break;
                    }
                    link_count = base_link_count + options.mutation_count;
                    target.capture_armed = 1U;
                    if (bpf_map_update_elem(
                            map_fd, &zero, &target, BPF_ANY) != 0) {
                        fprintf(stderr,
                            "runtime-state observer failed to configure maps\n");
                        collector.failed = true;
                        break;
                    }
                    mutations_armed = true;
                    if (fprintf(stderr,
                            "ARMED runtime_state mutations=%zu\n",
                            options.mutation_count) < 0
                        || fflush(stderr) != 0) {
                        collector.failed = true;
                        break;
                    }
                    continue;
                }
                if (strcmp(command, "SNAPSHOT\n") != 0) {
                    collector.failed = true;
                    break;
                }
                terminal.cutoff_monotonic_ns = monotonic_ns();
                if (terminal.cutoff_monotonic_ns == 0U
                    || !capture_stream_version_snapshot(
                        ring,
                        counter_fd,
                        stream_version_fd,
                        &collector,
                        &counters,
                        &stream_version)) {
                    (void)fprintf(stderr,
                        "DRAIN_FAILED runtime_state submitted=%" PRIu64
                        " received=%" PRIu64 " active=%" PRIu64 "\n",
                        (uint64_t)(counters.kernel_event_submit_count
                            + counters.state_unit_fragment_submit_count
                            + counters.state_unit_seal_submit_count),
                        (uint64_t)collector.native_event_receive_count,
                        (uint64_t)counters.active_emitter_count);
                    (void)fflush(stderr);
                    collector.failed = true;
                    break;
                }
                copy_counters(
                    &counters,
                    &collector,
                    &terminal,
                    terminal.cutoff_monotonic_ns);
                {
                    struct runtime_state_bpf_diagnostics diagnostics = {0};
                    if (bpf_map_lookup_elem(diagnostic_fd, &zero, &diagnostics) != 0
                        || fprintf(stderr,
                            "DIAGNOSTIC runtime_state candidate_missing=%" PRIu64
                            " shape_invalid=%" PRIu64
                            " data_read_failed=%" PRIu64
                            " type_read_failed=%" PRIu64
                            " data_null=%" PRIu64 "\n",
                            (uint64_t)diagnostics.candidate_missing_count,
                            (uint64_t)diagnostics.shape_invalid_count,
                            (uint64_t)diagnostics.tensor_data_read_failure_count,
                            (uint64_t)diagnostics.tensor_type_read_failure_count,
                            (uint64_t)diagnostics.tensor_data_null_count) < 0
                        || fflush(stderr) != 0) {
                        collector.failed = true;
                        break;
                    }
                    if (fprintf(stderr,
                            "APPLY_DIAGNOSTIC runtime_state state_conflict=%" PRIu64
                            " header_read_failed=%" PRIu64
                            " slot_shape_invalid=%" PRIu64
                            " batch_shape_invalid=%" PRIu64
                            " stream_count_zero=%" PRIu64
                            " stream_count_exceeds=%" PRIu64
                            " index_vector_extent_invalid=%" PRIu64
                            " token_count_zero=%" PRIu64
                            " token_count_exceeds=%" PRIu64
                            " token_count_exceeds_observed=%" PRIu64
                            " token_count_exceeds_streams=%" PRIu64
                            " token_stream_remainder=%" PRIu64
                            " token_read_failed=%" PRIu64
                            " token_shape_invalid=%" PRIu64
                            " return_unmatched=%" PRIu64
                            " state_delete_failed=%" PRIu64
                            " process_state_conflict=%" PRIu64
                            " process_return_unmatched=%" PRIu64
                            " process_apply_missing=%" PRIu64
                            " process_graph_compute_failed=%" PRIu64 "\n",
                            (uint64_t)diagnostics.apply_state_conflict_count,
                            (uint64_t)diagnostics.apply_header_read_failure_count,
                            (uint64_t)diagnostics.apply_slot_shape_invalid_count,
                            (uint64_t)diagnostics.apply_batch_shape_invalid_count,
                            (uint64_t)diagnostics.apply_stream_count_zero_count,
                            (uint64_t)diagnostics.apply_stream_count_exceeds_count,
                            (uint64_t)diagnostics.apply_index_vector_extent_invalid_count,
                            (uint64_t)diagnostics.apply_token_count_zero_count,
                            (uint64_t)diagnostics.apply_token_count_exceeds_count,
                            (uint64_t)(diagnostics.apply_token_count_exceeds_sample >> 32U),
                            (uint64_t)(diagnostics.apply_token_count_exceeds_sample & UINT32_MAX),
                            (uint64_t)diagnostics.apply_token_stream_remainder_count,
                            (uint64_t)diagnostics.apply_token_read_failure_count,
                            (uint64_t)diagnostics.apply_token_shape_invalid_count,
                            (uint64_t)diagnostics.apply_return_unmatched_count,
                            (uint64_t)diagnostics.apply_state_delete_failure_count,
                            (uint64_t)diagnostics.process_state_conflict_count,
                            (uint64_t)diagnostics.process_return_unmatched_count,
                            (uint64_t)diagnostics.process_apply_missing_count,
                            (uint64_t)diagnostics.process_graph_compute_failed_count) < 0
                        || fflush(stderr) != 0) {
                        collector.failed = true;
                        break;
                    }
                }
                if (!write_stream_version_frame(&collector, &stream_version)
                    || !write_counter_frame(&collector, &terminal, false)) {
                    collector.failed = true;
                    break;
                }
            }
            if ((input.revents & POLLHUP) != 0) {
                break;
            }
        }
    }
    if (collector.failed) {
        (void)report_gather_failure(&collector);
    }
    terminal.cutoff_monotonic_ns = monotonic_ns();
    if (terminal.cutoff_monotonic_ns == 0U
        || collector.failed
        || !capture_stream_version_snapshot(
            ring,
            counter_fd,
            stream_version_fd,
            &collector,
            &counters,
            &stream_version)) {
        goto cleanup;
    }
    /*
     * Keep return probes attached until the terminal drain observes every
     * call that entered before the completion boundary.  Detaching a
     * uretprobe first can strand inflight_call_count when that call returns.
     */
    for (index = 0U; index < link_count; ++index) {
        bpf_link__destroy(links[index]);
        links[index] = NULL;
    }
    copy_counters(
        &counters,
        &collector,
        &terminal,
        terminal.cutoff_monotonic_ns);
    if ((options.capture_mode == OBSERVER_CAPTURE_GATHER_WRITER_BOUNDARY
            && !report_bulk_metrics(&collector))
        || !write_stream_version_frame(&collector, &stream_version)
        || !write_counter_frame(&collector, &terminal, true)) {
        goto cleanup;
    }
    result = !mutations_armed
        || collector.failed
        || terminal.kernel_event_submit_count != terminal.native_event_receive_count
        || (options.capture_mode
                != OBSERVER_CAPTURE_GATHER_WRITER_BOUNDARY
            && terminal.native_event_receive_count
                != terminal.native_wire_frame_count)
        || terminal.ringbuf_reserve_fail_count != 0U
        || terminal.user_memory_read_failure_count != 0U
        || terminal.unmatched_call_count != 0U
        || terminal.unsupported_mutation_count != 0U
        || terminal.mapping_mutation_count != 0U
        || terminal.final_inflight_count != 0U
        || terminal.active_emitter_count != 0U
        || collector.pending_capture_count != 0U
        || collector.gathered_capture_count != 0U;

cleanup:
    if (options.capture_mode == OBSERVER_CAPTURE_GATHER_WRITER_BOUNDARY
        && options.target_tgid != 0U) {
        (void)kill((pid_t)options.target_tgid, SIGCONT);
        collector.target_stopped = false;
    }
    ring_buffer__free(ring);
    for (index = 0U; index < link_count; ++index) {
        bpf_link__destroy(links[index]);
    }
    bpf_object__close(object);
    for (index = 0U; index < RUNTIME_STATE_GATHER_MAX_PENDING; ++index) {
        free(collector.pending_captures[index].read_plans);
    }
    clear_gathered_captures(&collector);
    return result;
}
