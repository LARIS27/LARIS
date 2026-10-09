#include <linux/bpf.h>
#include <linux/ptrace.h>
#include <stdbool.h>

#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#include "runtime_state_bpf_shared.h"

#ifndef SIGSTOP
#define SIGSTOP 19
#endif

#if RUNTIME_STATE_MAX_APPLY_STREAMS != 8U
#error "stream-index verifier bound requires exactly eight apply streams"
#endif

#if (RUNTIME_STATE_MAX_APPLY_STREAMS & \
        (RUNTIME_STATE_MAX_APPLY_STREAMS - 1U)) != 0U
#error "stream-index verifier mask requires a power-of-two stream ceiling"
#endif

#if (RUNTIME_STATE_MAX_BATCH_FRAGMENTS & \
        (RUNTIME_STATE_MAX_BATCH_FRAGMENTS - 1U)) != 0U
#error "fragment-batch verifier mask requires a power-of-two ceiling"
#endif

struct constructor_state {
    __u64 cache_object;
    __u64 hparams;
    __u64 started_ns;
    __u32 tgid;
    __u32 tid;
    __u32 kv_size;
    __u32 n_seq_max;
    __u32 n_stream;
    __u32 n_pos_per_embd;
    __u32 v_trans;
    __u32 tensor_count;
    __u32 poisoned;
};

struct n_pos_call_state {
    __u64 hparams;
};

struct view_call_state {
    __u64 source_tensor_pointer;
    __u64 view_offset;
    __u64 ne0;
    __u64 ne1;
    __u64 nb1;
};

struct tensor_candidate_key {
    __u32 tgid;
    __u32 ordinal;
};

struct tensor_candidate {
    __u64 tensor_pointer;
    __u64 source_tensor_pointer;
    __u64 view_offset;
    __u64 row_size;
    __u64 cell_capacity;
    __u64 embedding_width;
};

struct buffer_call_state {
    __u64 buffer_handle;
    __u32 tgid;
    __u32 tid;
};

struct buffer_candidate {
    __u64 base_address;
    __u64 size_bytes;
    __u32 tgid;
    __u32 tid;
    __u32 emitted;
    __u32 reserved;
};

struct apply_call_state {
    __u64 apply_call_id;
    __u32 stream_mask;
    __u32 token_cell_count;
    __u64 prior_stream_versions[RUNTIME_STATE_MAX_APPLY_STREAMS];
    __u64 first_token_positions[RUNTIME_STATE_MAX_APPLY_STREAMS];
    __u64 first_cell_indices[RUNTIME_STATE_MAX_APPLY_STREAMS];
    __u32 sequence_ids[RUNTIME_STATE_MAX_APPLY_STREAMS];
    __u32 stream_token_counts[RUNTIME_STATE_MAX_APPLY_STREAMS];
};

struct apply_token_loop_context {
    __u64 pid_tgid;
    __u64 apply_call_id;
    __u64 stream_begin;
    __u64 outer_idxs_begin;
    __u64 positions;
    __u64 sequence_counts;
    __u64 sequences;
    __u32 n_stream_in_call;
    __u32 capture_mode;
    __u32 failed;
    __u32 reserved;
};

struct apply_token_plan_key {
    __u64 apply_call_id;
    __u32 token_ordinal;
    __u32 reserved;
};

struct apply_token_plan_loop_context {
    __u64 pid_tgid;
    __u64 apply_call_id;
    __u32 token_cell_count;
    __u32 emit;
    __u32 failed;
    __u32 reserved;
};

struct layout_bootstrap_state {
    __u64 phase;
};

struct bootstrap_tensor {
    __u64 tensor_pointer;
    __u64 tensor_address;
    __u64 row_size;
    __u64 cell_capacity;
    __u64 embedding_width;
    __u64 storage_size;
    __u32 ggml_type_id;
};

struct constructor_layout_loop_context {
    __u32 tgid;
    __u32 tid;
    __u32 n_stream;
    __u32 tensor_data_offset;
    __u32 tensor_type_offset;
    __u32 failed;
};

struct bootstrap_layout_loop_context {
    __u64 pid_tgid;
    __u64 layers_begin;
    __u64 expected_cell_capacity;
    __u32 n_layer;
    __u32 n_stream;
    __u32 failed;
    __u32 reserved;
};

struct state_unit_copy_loop_context {
    __u64 apply_call_id;
    __u64 writer_monotonic_ns;
    __u64 copy_byte_count;
    __u32 expected_n_stream;
    __u32 expected_n_layer;
    __u32 v_trans;
    __u32 fragment_index;
    __u32 plan_count;
    __u32 failed;
};

struct read_plan_schedule_context {
    __u64 apply_call_id;
    __u64 writer_monotonic_ns;
    __u64 first_token_position;
    __u64 first_cell_index;
    __u64 per_layer_copy_byte_count;
    __u32 expected_n_stream;
    __u32 expected_n_layer;
    __u32 row_size;
    __u32 cell_capacity;
    __u32 element_size;
    __u32 embedding_width;
    __u32 fragment_base;
    __u32 seq_id;
    __u32 stream_index;
    __u32 tensor_kind;
    __u32 kind_offset;
    __u32 token_count;
    __u32 transposed_value;
    __u32 chunks_per_dimension;
    __u32 fragments_per_tensor;
    __u32 failed;
};

struct flat_tensor_fragment_context {
    __u64 apply_call_id;
    __u64 writer_monotonic_ns;
    __u64 first_token_position;
    __u32 expected_n_stream;
    __u32 expected_n_layer;
    __u32 row_size;
    __u32 cell_capacity;
    __u32 element_size;
    __u32 embedding_width;
    __u32 first_cell_index;
    __u32 fragment_base;
    __u32 seq_id;
    __u32 stream_index;
    __u32 tensor_kind;
    __u32 kind_offset;
    __u32 token_count;
    __u32 transposed_value;
    __u32 chunks_per_dimension;
    __u32 fragments_per_tensor;
    __u32 batch_record_count;
    __u32 batch_payload_size;
    __u32 failed;
};

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct runtime_state_target_config);
} runtime_state_target SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 64);
    __type(key, __u64);
    __type(value, struct constructor_state);
} constructors_by_thread SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 16);
    __type(key, __u64);
    __type(value, struct constructor_state);
} completed_constructors_by_hparams SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 64);
    __type(key, __u64);
    __type(value, struct n_pos_call_state);
} n_pos_calls_by_thread SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 16);
    __type(key, __u64);
    __type(value, __u32);
} n_pos_by_hparams SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 64);
    __type(key, __u64);
    __type(value, struct view_call_state);
} views_by_thread SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, RUNTIME_STATE_MAX_TENSOR_EVENTS);
    __type(key, struct tensor_candidate_key);
    __type(value, struct tensor_candidate);
} tensor_candidates SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 64);
    __type(key, __u64);
    __type(value, struct buffer_call_state);
} buffer_base_calls SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 64);
    __type(key, __u64);
    __type(value, struct buffer_call_state);
} buffer_size_calls SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 64);
    __type(key, __u64);
    __type(value, struct buffer_candidate);
} buffers_by_handle SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, __u64);
    __type(value, struct apply_call_state);
} apply_calls_by_thread SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, __u64);
    __type(value, struct runtime_state_compute_call);
} compute_calls_by_thread SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, __u64);
    __type(value, __u32);
} ignored_compute_returns_by_thread SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 8192);
    __type(key, struct apply_token_plan_key);
    __type(value, struct runtime_state_kernel_token_cell_event);
} apply_token_plan SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, RUNTIME_STATE_MAX_APPLY_STREAMS);
    __type(key, __u32);
    __type(value, __u32);
} observed_sequence_stream SEC(".maps");

/*
 * Each stream is a disjoint Simple-KV slot partition.  The high bit marks a
 * writer in flight; the remaining bits carry the monotone apply_call_id.
 * Userspace samples these values directly before and after copying memory,
 * so correctness does not depend on ring-buffer delivery latency.
 */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, RUNTIME_STATE_MAX_APPLY_STREAMS);
    __type(key, __u32);
    __type(value, __u64);
} runtime_state_stream_versions SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct runtime_state_bpf_counters);
} runtime_state_counters SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct runtime_state_bpf_diagnostics);
} runtime_state_diagnostics SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct layout_bootstrap_state);
} runtime_state_layout_bootstrap SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct runtime_state_kernel_topology_event);
} runtime_state_retained_topology SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, RUNTIME_STATE_MAX_TENSOR_EVENTS);
    __type(key, __u32);
    __type(value, struct runtime_state_kernel_tensor_layout_event);
} runtime_state_retained_tensor_layouts SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 4U * 1024U * 1024U);
} runtime_state_events SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 16U * 1024U * 1024U);
} runtime_state_payload_events SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct runtime_state_kernel_state_unit_fragment_event);
} runtime_state_fragment_scratch SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct runtime_state_kernel_state_unit_schedule_batch_event);
} runtime_state_payload_scratch SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct runtime_state_kernel_event);
} runtime_state_event_scratch SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct apply_call_state);
} runtime_state_apply_call_scratch SEC(".maps");

static __always_inline void increment(__u64 *value) {
    (void)__sync_fetch_and_add(value, 1U);
}

static __always_inline struct runtime_state_target_config *target_config(__u64 pid_tgid) {
    __u32 key = 0U;
    struct runtime_state_target_config *config = bpf_map_lookup_elem(&runtime_state_target, &key);
    if (config == NULL || config->target_tgid == 0U || (__u32)(pid_tgid >> 32U) != config->target_tgid) {
        return NULL;
    }
    return config;
}

static __always_inline struct runtime_state_bpf_counters *global_counters(void) {
    __u32 key = 0U;
    return bpf_map_lookup_elem(&runtime_state_counters, &key);
}

static __always_inline struct runtime_state_bpf_diagnostics *global_diagnostics(void) {
    __u32 key = 0U;
    return bpf_map_lookup_elem(&runtime_state_diagnostics, &key);
}

static __always_inline struct runtime_state_kernel_event *global_event_scratch(void) {
    __u32 key = 0U;
    return bpf_map_lookup_elem(&runtime_state_event_scratch, &key);
}

static __always_inline struct apply_call_state *global_apply_call_scratch(void) {
    __u32 key = 0U;
    return bpf_map_lookup_elem(&runtime_state_apply_call_scratch, &key);
}

static __always_inline __u64 next_event_id(struct runtime_state_bpf_counters *counters) {
    return __sync_fetch_and_add(&counters->next_event_id, 1U) + 1U;
}

static __always_inline bool submit_event(
    const struct runtime_state_kernel_event *source,
    struct runtime_state_bpf_counters *counters,
    __u64 *type_counter
) {
    __u64 now;
    increment(&counters->active_emitter_count);
    struct runtime_state_kernel_event *destination = bpf_ringbuf_reserve(
        &runtime_state_events, sizeof(*destination), 0U);
    if (destination == NULL) {
        increment(&counters->ringbuf_reserve_fail_count);
        (void)__sync_fetch_and_sub(&counters->active_emitter_count, 1U);
        return false;
    }
    __builtin_memcpy(destination, source, sizeof(*destination));
    now = bpf_ktime_get_ns();
    switch (destination->event_type) {
        case RUNTIME_STATE_KERNEL_TOPOLOGY:
            destination->payload.topology.monotonic_ns = now;
            break;
        case RUNTIME_STATE_KERNEL_TENSOR_LAYOUT:
            destination->payload.tensor.monotonic_ns = now;
            break;
        case RUNTIME_STATE_KERNEL_KV_BACKING:
            destination->payload.backing.monotonic_ns = now;
            break;
        case RUNTIME_STATE_KERNEL_TOKEN_CELL:
            destination->payload.token.monotonic_ns = now;
            break;
        case RUNTIME_STATE_KERNEL_TOKEN_CELL_RANGE:
            destination->payload.token_range.monotonic_ns = now;
            break;
        case RUNTIME_STATE_KERNEL_APPLY_COMPLETE:
            destination->payload.apply_complete.monotonic_ns = now;
            break;
        case RUNTIME_STATE_KERNEL_MUTATION:
            destination->payload.mutation.monotonic_ns = now;
            break;
        default:
            bpf_ringbuf_discard(destination, 0U);
            increment(&counters->unsupported_mutation_count);
            (void)__sync_fetch_and_sub(&counters->active_emitter_count, 1U);
            return false;
    }
    bpf_ringbuf_submit(destination, 0U);
    if (type_counter != NULL) {
        increment(type_counter);
    }
    increment(&counters->kernel_event_submit_count);
    (void)__sync_fetch_and_sub(&counters->active_emitter_count, 1U);
    return true;
}

static __always_inline void emit_mutation(
    __u64 pid_tgid,
    __u32 mutation_kind,
    struct runtime_state_bpf_counters *counters
) {
    struct runtime_state_kernel_event *output;
    if (mutation_kind < RUNTIME_STATE_MUTATION_CACHE_SHIFT
        || mutation_kind > RUNTIME_STATE_MUTATION_SPECULATIVE_CONTEXT) {
        increment(&counters->unsupported_mutation_count);
        return;
    }
    output = global_event_scratch();
    if (output == NULL) {
        increment(&counters->unsupported_mutation_count);
        return;
    }
    __builtin_memset(output, 0, sizeof(*output));
    output->event_type = RUNTIME_STATE_KERNEL_MUTATION;
    output->payload.mutation.event_id = next_event_id(counters);
    output->payload.mutation.tgid = (__u32)(pid_tgid >> 32U);
    output->payload.mutation.tid = (__u32)pid_tgid;
    output->payload.mutation.mutation_kind = mutation_kind;
    (void)submit_event(output, counters, &counters->mutation_event_count);
}

SEC("uprobe")
int observe_forbidden_mutation(struct pt_regs *ctx) {
    const __u64 pid_tgid = bpf_get_current_pid_tgid();
    struct runtime_state_bpf_counters *counters;
    if (target_config(pid_tgid) == NULL) {
        return 0;
    }
    counters = global_counters();
    if (counters != NULL) {
        emit_mutation(pid_tgid, (__u32)bpf_get_attach_cookie(ctx), counters);
    }
    return 0;
}

static __always_inline void record_read_failure(struct runtime_state_bpf_counters *counters) {
    increment(&counters->user_memory_read_failure_count);
}

static __always_inline bool retain_topology(
    const struct runtime_state_kernel_topology_event *topology
);

static __always_inline bool retain_tensor_layout(
    __u32 ordinal,
    const struct runtime_state_kernel_tensor_layout_event *layout
);

static __always_inline void emit_constructor_layout(
    const struct constructor_state *constructor,
    const struct runtime_state_target_config *config,
    struct runtime_state_bpf_counters *counters
);

SEC("uprobe")
int observe_n_pos_per_embd_entry(struct pt_regs *ctx) {
    const __u64 pid_tgid = bpf_get_current_pid_tgid();
    struct runtime_state_bpf_counters *counters;
    struct n_pos_call_state state = {.hparams = PT_REGS_PARM1(ctx)};
    if (target_config(pid_tgid) == NULL) {
        return 0;
    }
    counters = global_counters();
    if (counters == NULL || state.hparams == 0U
        || bpf_map_update_elem(&n_pos_calls_by_thread, &pid_tgid, &state, BPF_NOEXIST) != 0) {
        if (counters != NULL) {
            increment(&counters->unmatched_call_count);
        }
        return 0;
    }
    increment(&counters->inflight_call_count);
    return 0;
}

SEC("uretprobe")
int observe_n_pos_per_embd_return(struct pt_regs *ctx) {
    const __u64 pid_tgid = bpf_get_current_pid_tgid();
    struct runtime_state_target_config *config = target_config(pid_tgid);
    struct runtime_state_bpf_counters *counters;
    struct n_pos_call_state *call;
    struct constructor_state *active_constructor;
    struct constructor_state *completed_constructor;
    __u32 observed = (__u32)PT_REGS_RC(ctx);
    if (config == NULL) {
        return 0;
    }
    counters = global_counters();
    call = bpf_map_lookup_elem(&n_pos_calls_by_thread, &pid_tgid);
    if (counters == NULL || call == NULL) {
        if (counters != NULL) {
            increment(&counters->unmatched_call_count);
        }
        return 0;
    }
    if (observed == 0U || observed > 4U
        || bpf_map_update_elem(&n_pos_by_hparams, &call->hparams, &observed, BPF_ANY) != 0) {
        increment(&counters->unsupported_mutation_count);
    } else {
        active_constructor = bpf_map_lookup_elem(&constructors_by_thread, &pid_tgid);
        if (active_constructor != NULL && active_constructor->hparams == call->hparams) {
            active_constructor->n_pos_per_embd = observed;
        }
        completed_constructor = bpf_map_lookup_elem(
            &completed_constructors_by_hparams, &call->hparams);
        if (completed_constructor != NULL) {
            completed_constructor->n_pos_per_embd = observed;
            emit_constructor_layout(completed_constructor, config, counters);
            (void)bpf_map_delete_elem(
                &completed_constructors_by_hparams, &call->hparams);
            (void)__sync_fetch_and_sub(&counters->inflight_call_count, 1U);
        }
    }
    (void)bpf_map_delete_elem(&n_pos_calls_by_thread, &pid_tgid);
    (void)__sync_fetch_and_sub(&counters->inflight_call_count, 1U);
    return 0;
}

SEC("uprobe")
int observe_kv_constructor_entry(struct pt_regs *ctx) {
    const __u64 pid_tgid = bpf_get_current_pid_tgid();
    struct runtime_state_target_config *config = target_config(pid_tgid);
    struct runtime_state_bpf_counters *counters;
    struct constructor_state state = {0};
    __u32 *observed_n_pos;
    __u64 stack_pointer;
    __u8 unified = 0U;
    if (config == NULL) {
        return 0;
    }
    counters = global_counters();
    if (counters == NULL) {
        return 0;
    }
    if (bpf_map_lookup_elem(&constructors_by_thread, &pid_tgid) != NULL) {
        increment(&counters->unmatched_call_count);
        return 0;
    }
    stack_pointer = PT_REGS_SP(ctx);
    state.cache_object = PT_REGS_PARM1(ctx);
    state.hparams = PT_REGS_PARM3(ctx);
    state.started_ns = bpf_ktime_get_ns();
    state.tgid = (__u32)(pid_tgid >> 32U);
    state.tid = (__u32)pid_tgid;
    state.v_trans = (__u32)(PT_REGS_PARM6(ctx) & 1U);
    if (bpf_probe_read_user(&unified, sizeof(unified), (const void *)(stack_pointer + 16U)) != 0
        || bpf_probe_read_user(&state.kv_size, sizeof(state.kv_size), (const void *)(stack_pointer + 24U)) != 0
        || bpf_probe_read_user(&state.n_seq_max, sizeof(state.n_seq_max), (const void *)(stack_pointer + 32U)) != 0) {
        record_read_failure(counters);
        return 0;
    }
    state.n_stream = unified != 0U ? 1U : state.n_seq_max;
    observed_n_pos = bpf_map_lookup_elem(&n_pos_by_hparams, &state.hparams);
    if (observed_n_pos != NULL) {
        state.n_pos_per_embd = *observed_n_pos;
    }
    if (state.cache_object == 0U || state.hparams == 0U
        || state.kv_size == 0U || state.n_seq_max != config->expected_n_seq_max
        || state.n_stream != config->expected_n_stream || state.v_trans > 1U) {
        increment(&counters->unsupported_mutation_count);
        return 0;
    }
    if (bpf_map_update_elem(&constructors_by_thread, &pid_tgid, &state, BPF_NOEXIST) != 0) {
        increment(&counters->unmatched_call_count);
        return 0;
    }
    increment(&counters->inflight_call_count);
    return 0;
}

SEC("uprobe")
int observe_view_2d_entry(struct pt_regs *ctx) {
    const __u64 pid_tgid = bpf_get_current_pid_tgid();
    struct constructor_state *constructor;
    struct view_call_state state = {0};
    if (target_config(pid_tgid) == NULL) {
        return 0;
    }
    constructor = bpf_map_lookup_elem(&constructors_by_thread, &pid_tgid);
    if (constructor == NULL || constructor->poisoned != 0U) {
        return 0;
    }
    state.source_tensor_pointer = PT_REGS_PARM2(ctx);
    state.view_offset = PT_REGS_PARM6(ctx);
    state.ne0 = PT_REGS_PARM3(ctx);
    state.ne1 = PT_REGS_PARM4(ctx);
    state.nb1 = PT_REGS_PARM5(ctx);
    if (state.source_tensor_pointer == 0U || state.ne0 == 0U
        || state.ne1 != constructor->kv_size || state.nb1 == 0U
        || constructor->tensor_count >= RUNTIME_STATE_MAX_TENSOR_EVENTS) {
        constructor->poisoned = 1U;
        return 0;
    }
    (void)bpf_map_update_elem(&views_by_thread, &pid_tgid, &state, BPF_ANY);
    return 0;
}

SEC("uretprobe")
int observe_view_2d_return(struct pt_regs *ctx) {
    const __u64 pid_tgid = bpf_get_current_pid_tgid();
    struct constructor_state *constructor;
    struct view_call_state *view;
    struct tensor_candidate candidate = {0};
    struct tensor_candidate_key key = {0};
    if (target_config(pid_tgid) == NULL) {
        return 0;
    }
    constructor = bpf_map_lookup_elem(&constructors_by_thread, &pid_tgid);
    view = bpf_map_lookup_elem(&views_by_thread, &pid_tgid);
    if (constructor == NULL || view == NULL || constructor->poisoned != 0U) {
        return 0;
    }
    candidate.tensor_pointer = PT_REGS_RC(ctx);
    candidate.source_tensor_pointer = view->source_tensor_pointer;
    candidate.view_offset = view->view_offset;
    candidate.row_size = view->nb1;
    candidate.cell_capacity = view->ne1;
    candidate.embedding_width = view->ne0;
    key.tgid = constructor->tgid;
    key.ordinal = constructor->tensor_count;
    if (candidate.tensor_pointer == 0U
        || bpf_map_update_elem(&tensor_candidates, &key, &candidate, BPF_NOEXIST) != 0) {
        constructor->poisoned = 1U;
    } else {
        constructor->tensor_count += 1U;
    }
    (void)bpf_map_delete_elem(&views_by_thread, &pid_tgid);
    return 0;
}

static long emit_constructor_tensor(__u32 index, void *data) {
    struct constructor_layout_loop_context *loop = data;
    struct runtime_state_bpf_counters *counters = global_counters();
    struct runtime_state_bpf_diagnostics *diagnostics = global_diagnostics();
    struct tensor_candidate_key key;
    struct tensor_candidate *candidate;
    struct runtime_state_kernel_event output = {0};
    __u64 tensor_address = 0U;
    __u64 tensor_data_field;
    __u64 tensor_type_field;
    __u32 ggml_type_id = 0U;
    if (loop == NULL || counters == NULL || loop->n_stream == 0U) {
        if (loop != NULL) {
            loop->failed = 1U;
        }
        return 1;
    }
    key.tgid = loop->tgid;
    key.ordinal = index;
    candidate = bpf_map_lookup_elem(&tensor_candidates, &key);
    if (candidate == NULL) {
        if (diagnostics != NULL) {
            increment(&diagnostics->candidate_missing_count);
        }
        record_read_failure(counters);
        return 0;
    }
    if (candidate->embedding_width == 0U
        || candidate->row_size % candidate->embedding_width != 0U) {
        if (diagnostics != NULL) {
            increment(&diagnostics->shape_invalid_count);
        }
        record_read_failure(counters);
        (void)bpf_map_delete_elem(&tensor_candidates, &key);
        return 0;
    }
    if (candidate->tensor_pointer > (~(__u64)0) - loop->tensor_data_offset
        || candidate->tensor_pointer > (~(__u64)0) - loop->tensor_type_offset) {
        if (diagnostics != NULL) {
            increment(&diagnostics->shape_invalid_count);
        }
        record_read_failure(counters);
        (void)bpf_map_delete_elem(&tensor_candidates, &key);
        return 0;
    }
    tensor_data_field = candidate->tensor_pointer + loop->tensor_data_offset;
    tensor_type_field = candidate->tensor_pointer + loop->tensor_type_offset;
    if (bpf_probe_read_user(&tensor_address, sizeof(tensor_address),
            (const void *)tensor_data_field) != 0) {
        if (diagnostics != NULL) {
            increment(&diagnostics->tensor_data_read_failure_count);
        }
        record_read_failure(counters);
        (void)bpf_map_delete_elem(&tensor_candidates, &key);
        return 0;
    }
    if (tensor_address == 0U) {
        __u64 source_data = 0U;
        __u64 source_data_field;
        if (candidate->source_tensor_pointer == 0U
            || candidate->source_tensor_pointer
                > (~(__u64)0) - loop->tensor_data_offset) {
            if (diagnostics != NULL) {
                increment(&diagnostics->tensor_data_read_failure_count);
            }
            record_read_failure(counters);
            (void)bpf_map_delete_elem(&tensor_candidates, &key);
            return 0;
        }
        source_data_field = candidate->source_tensor_pointer
            + loop->tensor_data_offset;
        if (bpf_probe_read_user(&source_data, sizeof(source_data),
                (const void *)source_data_field) != 0) {
            if (diagnostics != NULL) {
                increment(&diagnostics->tensor_data_read_failure_count);
            }
            record_read_failure(counters);
            (void)bpf_map_delete_elem(&tensor_candidates, &key);
            return 0;
        }
        if (source_data == 0U) {
            if (diagnostics != NULL) {
                increment(&diagnostics->tensor_data_null_count);
            }
            record_read_failure(counters);
            (void)bpf_map_delete_elem(&tensor_candidates, &key);
            return 0;
        }
        if (candidate->view_offset > (~(__u64)0) - source_data) {
            if (diagnostics != NULL) {
                increment(&diagnostics->shape_invalid_count);
            }
            record_read_failure(counters);
            (void)bpf_map_delete_elem(&tensor_candidates, &key);
            return 0;
        }
        tensor_address = source_data + candidate->view_offset;
    }
    if (bpf_probe_read_user(&ggml_type_id, sizeof(ggml_type_id),
            (const void *)tensor_type_field) != 0) {
        if (diagnostics != NULL) {
            increment(&diagnostics->tensor_type_read_failure_count);
        }
        record_read_failure(counters);
        (void)bpf_map_delete_elem(&tensor_candidates, &key);
        return 0;
    }
    output.event_type = RUNTIME_STATE_KERNEL_TENSOR_LAYOUT;
    output.payload.tensor.event_id = next_event_id(counters);
    output.payload.tensor.tgid = loop->tgid;
    output.payload.tensor.tid = loop->tid;
    output.payload.tensor.monotonic_ns = bpf_ktime_get_ns();
    output.payload.tensor.kind = runtime_state_tensor_kind_for_ordinal(
        index, loop->n_stream);
    output.payload.tensor.layer = runtime_state_tensor_layer_for_ordinal(
        index, loop->n_stream);
    output.payload.tensor.stream_index = runtime_state_tensor_stream_for_ordinal(
        index, loop->n_stream);
    output.payload.tensor.ggml_type_id = ggml_type_id;
    output.payload.tensor.tensor_address = tensor_address;
    output.payload.tensor.row_size = candidate->row_size;
    output.payload.tensor.cell_capacity = candidate->cell_capacity;
    output.payload.tensor.element_size =
        candidate->row_size / candidate->embedding_width;
    output.payload.tensor.embedding_width = candidate->embedding_width;
    (void)bpf_map_delete_elem(&tensor_candidates, &key);
    if (!retain_tensor_layout(index, &output.payload.tensor)) {
        increment(&counters->unsupported_mutation_count);
        return 0;
    }
    (void)submit_event(&output, counters, &counters->tensor_layout_event_count);
    return 0;
}

static __always_inline void emit_constructor_layout(
    const struct constructor_state *constructor,
    const struct runtime_state_target_config *config,
    struct runtime_state_bpf_counters *counters
) {
    const __u32 tensors_per_layer = 2U * constructor->n_stream;
    __u32 observed_n_layer;
    struct runtime_state_kernel_event output = {0};
    struct constructor_layout_loop_context loop = {0};
    long processed;
    if (constructor->poisoned != 0U || tensors_per_layer == 0U
        || constructor->tensor_count == 0U
        || constructor->tensor_count % tensors_per_layer != 0U) {
        increment(&counters->unsupported_mutation_count);
        return;
    }
    observed_n_layer = constructor->tensor_count / tensors_per_layer;
    if (observed_n_layer == 0U
        || observed_n_layer > RUNTIME_STATE_MAX_TENSOR_EVENTS
            / (2U * constructor->n_stream)
        || (config->expected_n_layer != 0U
            && observed_n_layer != config->expected_n_layer)) {
        increment(&counters->unsupported_mutation_count);
        return;
    }
    output.event_type = RUNTIME_STATE_KERNEL_TOPOLOGY;
    output.payload.topology.event_id = next_event_id(counters);
    output.payload.topology.tgid = constructor->tgid;
    output.payload.topology.tid = constructor->tid;
    output.payload.topology.monotonic_ns = bpf_ktime_get_ns();
    output.payload.topology.n_ctx = constructor->kv_size * constructor->n_stream;
    output.payload.topology.n_stream = constructor->n_stream;
    output.payload.topology.n_seq_max = constructor->n_seq_max;
    output.payload.topology.n_layer = observed_n_layer;
    output.payload.topology.v_trans = constructor->v_trans;
    output.payload.topology.n_pos_per_embd = constructor->n_pos_per_embd;
    if (!retain_topology(&output.payload.topology)) {
        increment(&counters->unsupported_mutation_count);
        return;
    }
    (void)submit_event(&output, counters, &counters->topology_event_count);

    loop.tgid = constructor->tgid;
    loop.tid = constructor->tid;
    loop.n_stream = constructor->n_stream;
    loop.tensor_data_offset = config->tensor_data_offset;
    loop.tensor_type_offset = config->tensor_type_offset;
    processed = bpf_loop(
        constructor->tensor_count, emit_constructor_tensor, &loop, 0U);
    if (processed != (long)constructor->tensor_count || loop.failed != 0U) {
        increment(&counters->unsupported_mutation_count);
    }
}

SEC("uretprobe")
int observe_kv_constructor_return(struct pt_regs *ctx) {
    const __u64 pid_tgid = bpf_get_current_pid_tgid();
    struct runtime_state_target_config *config = target_config(pid_tgid);
    struct runtime_state_bpf_counters *counters;
    struct constructor_state *constructor;
    if (config == NULL) {
        return 0;
    }
    counters = global_counters();
    constructor = bpf_map_lookup_elem(&constructors_by_thread, &pid_tgid);
    if (counters == NULL || constructor == NULL) {
        if (counters != NULL) {
            increment(&counters->unmatched_call_count);
        }
        return 0;
    }
    if (constructor->n_pos_per_embd == 0U) {
        __u32 *observed_n_pos = bpf_map_lookup_elem(
            &n_pos_by_hparams, &constructor->hparams);
        if (observed_n_pos != NULL) {
            constructor->n_pos_per_embd = *observed_n_pos;
        }
    }
    if (constructor->n_pos_per_embd == 0U) {
        if (bpf_map_update_elem(
                &completed_constructors_by_hparams,
                &constructor->hparams,
                constructor,
                BPF_NOEXIST) != 0) {
            increment(&counters->unmatched_call_count);
            (void)__sync_fetch_and_sub(&counters->inflight_call_count, 1U);
        }
        (void)bpf_map_delete_elem(&constructors_by_thread, &pid_tgid);
        return 0;
    }
    emit_constructor_layout(constructor, config, counters);
    (void)bpf_map_delete_elem(&constructors_by_thread, &pid_tgid);
    (void)__sync_fetch_and_sub(&counters->inflight_call_count, 1U);
    (void)ctx;
    return 0;
}

static __always_inline bool prepare_buffer_call(
    struct pt_regs *ctx,
    __u64 pid_tgid,
    struct buffer_call_state *state
) {
    if (target_config(pid_tgid) == NULL || bpf_map_lookup_elem(&constructors_by_thread, &pid_tgid) == NULL) {
        return false;
    }
    state->buffer_handle = PT_REGS_PARM1(ctx);
    state->tgid = (__u32)(pid_tgid >> 32U);
    state->tid = (__u32)pid_tgid;
    return state->buffer_handle != 0U;
}

static __always_inline void emit_backing_if_complete(
    __u64 buffer_handle,
    struct buffer_candidate *candidate,
    struct runtime_state_bpf_counters *counters
) {
    struct runtime_state_kernel_event output = {0};
    if (candidate->emitted != 0U || candidate->base_address == 0U || candidate->size_bytes == 0U) {
        return;
    }
    output.event_type = RUNTIME_STATE_KERNEL_KV_BACKING;
    output.payload.backing.event_id = next_event_id(counters);
    output.payload.backing.tgid = candidate->tgid;
    output.payload.backing.tid = candidate->tid;
    output.payload.backing.monotonic_ns = bpf_ktime_get_ns();
    output.payload.backing.base_address = candidate->base_address;
    output.payload.backing.size_bytes = candidate->size_bytes;
    if (submit_event(&output, counters, &counters->kv_backing_event_count)) {
        candidate->emitted = 1U;
    }
    (void)buffer_handle;
}

SEC("uprobe")
int observe_buffer_base_entry(struct pt_regs *ctx) {
    const __u64 pid_tgid = bpf_get_current_pid_tgid();
    struct buffer_call_state state = {0};
    if (prepare_buffer_call(ctx, pid_tgid, &state)) {
        (void)bpf_map_update_elem(
            &buffer_base_calls, &pid_tgid, &state, BPF_ANY);
    }
    return 0;
}

SEC("uretprobe")
int observe_buffer_base_return(struct pt_regs *ctx) {
    const __u64 pid_tgid = bpf_get_current_pid_tgid();
    struct buffer_call_state *call = bpf_map_lookup_elem(&buffer_base_calls, &pid_tgid);
    struct buffer_candidate initial = {0};
    struct buffer_candidate *candidate;
    struct runtime_state_bpf_counters *counters = global_counters();
    if (call == NULL || counters == NULL) {
        return 0;
    }
    initial.tgid = call->tgid;
    initial.tid = call->tid;
    (void)bpf_map_update_elem(&buffers_by_handle, &call->buffer_handle, &initial, BPF_NOEXIST);
    candidate = bpf_map_lookup_elem(&buffers_by_handle, &call->buffer_handle);
    if (candidate != NULL) {
        candidate->base_address = PT_REGS_RC(ctx);
        emit_backing_if_complete(call->buffer_handle, candidate, counters);
    }
    (void)bpf_map_delete_elem(&buffer_base_calls, &pid_tgid);
    return 0;
}

SEC("uprobe")
int observe_buffer_size_entry(struct pt_regs *ctx) {
    const __u64 pid_tgid = bpf_get_current_pid_tgid();
    struct buffer_call_state state = {0};
    if (prepare_buffer_call(ctx, pid_tgid, &state)) {
        (void)bpf_map_update_elem(
            &buffer_size_calls, &pid_tgid, &state, BPF_ANY);
    }
    return 0;
}

SEC("uretprobe")
int observe_buffer_size_return(struct pt_regs *ctx) {
    const __u64 pid_tgid = bpf_get_current_pid_tgid();
    struct buffer_call_state *call = bpf_map_lookup_elem(&buffer_size_calls, &pid_tgid);
    struct buffer_candidate initial = {0};
    struct buffer_candidate *candidate;
    struct runtime_state_bpf_counters *counters = global_counters();
    if (call == NULL || counters == NULL) {
        return 0;
    }
    initial.tgid = call->tgid;
    initial.tid = call->tid;
    (void)bpf_map_update_elem(&buffers_by_handle, &call->buffer_handle, &initial, BPF_NOEXIST);
    candidate = bpf_map_lookup_elem(&buffers_by_handle, &call->buffer_handle);
    if (candidate != NULL) {
        candidate->size_bytes = PT_REGS_RC(ctx);
        emit_backing_if_complete(call->buffer_handle, candidate, counters);
    }
    (void)bpf_map_delete_elem(&buffer_size_calls, &pid_tgid);
    return 0;
}

static __always_inline bool resolve_bootstrap_tensor(
    __u64 layers_begin,
    __u32 n_layer,
    __u32 n_stream,
    __u32 ordinal,
    const struct runtime_state_target_config *config,
    struct bootstrap_tensor *result
) {
    const __u32 tensors_per_layer = 2U * n_stream;
    const __u32 layer_index = ordinal / tensors_per_layer;
    const __u32 stream_index = (ordinal % tensors_per_layer) / 2U;
    const __u32 kind = runtime_state_tensor_kind_for_ordinal(ordinal, n_stream);
    const __u32 vector_offset = kind == RUNTIME_STATE_TENSOR_KIND_K
        ? RUNTIME_STATE_KV_LAYER_K_STREAM_OFFSET
        : RUNTIME_STATE_KV_LAYER_V_STREAM_OFFSET;
    __u64 layer_address;
    __u64 vector_address;
    __u64 vector_begin = 0U;
    __u64 vector_end = 0U;
    __u64 tensor_pointer = 0U;
    __u64 tensor_address = 0U;
    __u64 view_source = 0U;
    __u64 view_offset = 0U;
    __u64 storage_size;
    if (result == NULL || config == NULL || n_stream == 0U
        || layer_index >= n_layer || stream_index >= n_stream
        || layers_begin > (~(__u64)0)
            - layer_index * RUNTIME_STATE_KV_LAYER_SIZE) {
        return false;
    }
    layer_address = layers_begin + layer_index * RUNTIME_STATE_KV_LAYER_SIZE;
    if (layer_address > (~(__u64)0) - vector_offset) {
        return false;
    }
    vector_address = layer_address + vector_offset;
    if (bpf_probe_read_user(&vector_begin, sizeof(vector_begin),
            (const void *)(vector_address + RUNTIME_STATE_VECTOR_BEGIN_OFFSET)) != 0
        || bpf_probe_read_user(&vector_end, sizeof(vector_end),
            (const void *)(vector_address + RUNTIME_STATE_VECTOR_END_OFFSET)) != 0
        || vector_begin == 0U
        || vector_end != vector_begin + n_stream * sizeof(__u64)
        || bpf_probe_read_user(&tensor_pointer, sizeof(tensor_pointer),
            (const void *)(vector_begin + stream_index * sizeof(__u64))) != 0
        || tensor_pointer == 0U) {
        return false;
    }
    if (bpf_probe_read_user(&result->ggml_type_id, sizeof(result->ggml_type_id),
            (const void *)(tensor_pointer + config->tensor_type_offset)) != 0
        || bpf_probe_read_user(&result->embedding_width,
            sizeof(result->embedding_width),
            (const void *)(tensor_pointer + RUNTIME_STATE_TENSOR_NE_OFFSET)) != 0
        || bpf_probe_read_user(&result->cell_capacity,
            sizeof(result->cell_capacity),
            (const void *)(tensor_pointer + RUNTIME_STATE_TENSOR_NE_OFFSET
                + sizeof(__u64))) != 0
        || bpf_probe_read_user(&result->row_size, sizeof(result->row_size),
            (const void *)(tensor_pointer + RUNTIME_STATE_TENSOR_NB_OFFSET
                + sizeof(__u64))) != 0
        || bpf_probe_read_user(&tensor_address, sizeof(tensor_address),
            (const void *)(tensor_pointer + config->tensor_data_offset)) != 0) {
        return false;
    }
    if (tensor_address == 0U) {
        if (bpf_probe_read_user(&view_source, sizeof(view_source),
                (const void *)(tensor_pointer
                    + RUNTIME_STATE_TENSOR_VIEW_SRC_OFFSET)) != 0
            || bpf_probe_read_user(&view_offset, sizeof(view_offset),
                (const void *)(tensor_pointer
                    + RUNTIME_STATE_TENSOR_VIEW_OFFS_OFFSET)) != 0
            || view_source == 0U
            || bpf_probe_read_user(&tensor_address, sizeof(tensor_address),
                (const void *)(view_source + config->tensor_data_offset)) != 0
            || tensor_address == 0U
            || tensor_address > (~(__u64)0) - view_offset) {
            return false;
        }
        tensor_address += view_offset;
    }
    if (result->embedding_width == 0U || result->cell_capacity == 0U
        || result->row_size == 0U
        || result->row_size % result->embedding_width != 0U
        || result->row_size > (~(__u32)0)
        || result->cell_capacity > (~(__u32)0)) {
        return false;
    }
    storage_size = result->row_size * result->cell_capacity;
    if (tensor_address > (~(__u64)0) - storage_size) {
        return false;
    }
    result->tensor_pointer = tensor_pointer;
    result->tensor_address = tensor_address;
    result->storage_size = storage_size;
    return true;
}

static long emit_bootstrap_tensor(__u32 ordinal, void *data) {
    struct bootstrap_layout_loop_context *loop = data;
    struct runtime_state_target_config *config;
    struct runtime_state_bpf_counters *counters = global_counters();
    struct bootstrap_tensor tensor = {0};
    struct runtime_state_kernel_event *output;
    __u32 kind;
    if (loop == NULL || counters == NULL || loop->n_stream == 0U) {
        if (loop != NULL) {
            loop->failed = 1U;
        }
        return 1;
    }
    config = target_config(loop->pid_tgid);
    output = global_event_scratch();
    if (config == NULL
        || output == NULL
        || !resolve_bootstrap_tensor(
            loop->layers_begin,
            loop->n_layer,
            loop->n_stream,
            ordinal,
            config,
            &tensor)
        || tensor.cell_capacity != loop->expected_cell_capacity) {
        record_read_failure(counters);
        increment(&counters->unsupported_mutation_count);
        loop->failed = 1U;
        return 1;
    }
    kind = runtime_state_tensor_kind_for_ordinal(ordinal, loop->n_stream);
    __builtin_memset(output, 0, sizeof(*output));
    output->event_type = RUNTIME_STATE_KERNEL_TENSOR_LAYOUT;
    output->payload.tensor.event_id = next_event_id(counters);
    output->payload.tensor.tgid = (__u32)(loop->pid_tgid >> 32U);
    output->payload.tensor.tid = (__u32)loop->pid_tgid;
    output->payload.tensor.monotonic_ns = bpf_ktime_get_ns();
    output->payload.tensor.kind = kind;
    output->payload.tensor.layer = runtime_state_tensor_layer_for_ordinal(
        ordinal, loop->n_stream);
    output->payload.tensor.stream_index =
        runtime_state_tensor_stream_for_ordinal(ordinal, loop->n_stream);
    output->payload.tensor.ggml_type_id = tensor.ggml_type_id;
    output->payload.tensor.tensor_address = tensor.tensor_address;
    output->payload.tensor.row_size = tensor.row_size;
    output->payload.tensor.cell_capacity = tensor.cell_capacity;
    output->payload.tensor.element_size =
        tensor.row_size / tensor.embedding_width;
    output->payload.tensor.embedding_width = tensor.embedding_width;
    if (!retain_tensor_layout(ordinal, &output->payload.tensor)) {
        increment(&counters->unsupported_mutation_count);
        loop->failed = 1U;
        return 1;
    }
    (void)submit_event(
        output, counters, &counters->tensor_layout_event_count);

    __builtin_memset(output, 0, sizeof(*output));
    output->event_type = RUNTIME_STATE_KERNEL_KV_BACKING;
    output->payload.backing.event_id = next_event_id(counters);
    output->payload.backing.tgid = (__u32)(loop->pid_tgid >> 32U);
    output->payload.backing.tid = (__u32)loop->pid_tgid;
    output->payload.backing.monotonic_ns = bpf_ktime_get_ns();
    output->payload.backing.base_address = tensor.tensor_address;
    output->payload.backing.size_bytes = tensor.storage_size;
    (void)submit_event(
        output, counters, &counters->kv_backing_event_count);
    return 0;
}

static __always_inline void bootstrap_layout_from_apply(
    __u64 pid_tgid,
    __u64 cache_object,
    const struct runtime_state_target_config *config,
    struct runtime_state_bpf_counters *counters
) {
    __u32 zero = 0U;
    struct layout_bootstrap_state *state;
    __u64 hparams = 0U;
    __u64 layers_begin = 0U;
    __u64 layers_end = 0U;
    __u32 n_stream = 0U;
    __u32 n_seq_max = 0U;
    __u32 n_layer = 0U;
    __u32 rope_type = 0U;
    __u32 n_pos_per_embd;
    __u8 v_trans = 0U;
    __u32 tensor_count;
    struct bootstrap_tensor first = {0};
    struct runtime_state_kernel_event *output;
    struct bootstrap_layout_loop_context loop = {0};
    long processed;
    if (cache_object == 0U || config == NULL || counters == NULL
        || counters->topology_event_count != 0U) {
        return;
    }
    state = bpf_map_lookup_elem(&runtime_state_layout_bootstrap, &zero);
    if (state == NULL
        || __sync_val_compare_and_swap(&state->phase, 0ULL, 1ULL) != 0ULL) {
        return;
    }
    output = global_event_scratch();
    if (output == NULL) {
        increment(&counters->unsupported_mutation_count);
        return;
    }
    if (bpf_probe_read_user(&hparams, sizeof(hparams),
            (const void *)(cache_object + RUNTIME_STATE_CACHE_HPARAMS_OFFSET)) != 0
        || bpf_probe_read_user(&v_trans, sizeof(v_trans),
            (const void *)(cache_object + RUNTIME_STATE_CACHE_V_TRANS_OFFSET)) != 0
        || bpf_probe_read_user(&n_seq_max, sizeof(n_seq_max),
            (const void *)(cache_object + RUNTIME_STATE_CACHE_N_SEQ_MAX_OFFSET)) != 0
        || bpf_probe_read_user(&n_stream, sizeof(n_stream),
            (const void *)(cache_object + RUNTIME_STATE_CACHE_N_STREAM_OFFSET)) != 0
        || bpf_probe_read_user(&layers_begin, sizeof(layers_begin),
            (const void *)(cache_object + RUNTIME_STATE_CACHE_LAYERS_OFFSET
                + RUNTIME_STATE_VECTOR_BEGIN_OFFSET)) != 0
        || bpf_probe_read_user(&layers_end, sizeof(layers_end),
            (const void *)(cache_object + RUNTIME_STATE_CACHE_LAYERS_OFFSET
                + RUNTIME_STATE_VECTOR_END_OFFSET)) != 0
        || hparams == 0U || v_trans > 1U
        || n_stream != config->expected_n_stream
        || n_seq_max != config->expected_n_seq_max
        || layers_begin == 0U || layers_end <= layers_begin
        || (layers_end - layers_begin) % RUNTIME_STATE_KV_LAYER_SIZE != 0U
        || bpf_probe_read_user(&rope_type, sizeof(rope_type),
            (const void *)(hparams + RUNTIME_STATE_HPARAMS_ROPE_TYPE_OFFSET)) != 0) {
        record_read_failure(counters);
        increment(&counters->unsupported_mutation_count);
        return;
    }
    n_layer = (__u32)((layers_end - layers_begin) / RUNTIME_STATE_KV_LAYER_SIZE);
    if (n_layer == 0U
        || n_layer > RUNTIME_STATE_MAX_TENSOR_EVENTS / (2U * n_stream)
        || (config->expected_n_layer != 0U
            && n_layer != config->expected_n_layer)
        || !resolve_bootstrap_tensor(
            layers_begin, n_layer, n_stream, 0U, config, &first)
        || first.cell_capacity > (~(__u32)0) / n_stream) {
        record_read_failure(counters);
        increment(&counters->unsupported_mutation_count);
        return;
    }
    n_pos_per_embd = (rope_type & ~32U) == 8U ? 4U : 1U;
    __builtin_memset(output, 0, sizeof(*output));
    output->event_type = RUNTIME_STATE_KERNEL_TOPOLOGY;
    output->payload.topology.event_id = next_event_id(counters);
    output->payload.topology.tgid = (__u32)(pid_tgid >> 32U);
    output->payload.topology.tid = (__u32)pid_tgid;
    output->payload.topology.monotonic_ns = bpf_ktime_get_ns();
    output->payload.topology.n_ctx = (__u32)first.cell_capacity * n_stream;
    output->payload.topology.n_stream = n_stream;
    output->payload.topology.n_seq_max = n_seq_max;
    output->payload.topology.n_layer = n_layer;
    output->payload.topology.v_trans = v_trans;
    output->payload.topology.n_pos_per_embd = n_pos_per_embd;
    if (!retain_topology(&output->payload.topology)) {
        increment(&counters->unsupported_mutation_count);
        return;
    }
    (void)submit_event(output, counters, &counters->topology_event_count);

    tensor_count = 2U * n_stream * n_layer;
    loop.pid_tgid = pid_tgid;
    loop.layers_begin = layers_begin;
    loop.expected_cell_capacity = first.cell_capacity;
    loop.n_layer = n_layer;
    loop.n_stream = n_stream;
    processed = bpf_loop(
        tensor_count, emit_bootstrap_tensor, &loop, 0U);
    if (processed != (long)tensor_count || loop.failed != 0U) {
        return;
    }
    state->phase = 2ULL;
}

SEC("uprobe")
int observe_apply_topology_bootstrap(struct pt_regs *ctx) {
    const __u64 pid_tgid = bpf_get_current_pid_tgid();
    struct runtime_state_target_config *config = target_config(pid_tgid);
    struct runtime_state_bpf_counters *counters;
    if (config == NULL) {
        return 0;
    }
    counters = global_counters();
    if (counters != NULL && counters->topology_event_count == 0U) {
        bootstrap_layout_from_apply(
            pid_tgid, PT_REGS_PARM1(ctx), config, counters);
    }
    return 0;
}

static __always_inline bool retain_topology(
    const struct runtime_state_kernel_topology_event *topology
) {
    const __u32 zero = 0U;
    return topology != NULL
        && bpf_map_update_elem(
            &runtime_state_retained_topology,
            &zero,
            topology,
            BPF_ANY) == 0;
}

static __always_inline bool retain_tensor_layout(
    __u32 ordinal,
    const struct runtime_state_kernel_tensor_layout_event *layout
) {
    __u64 storage_size;
    if (layout == NULL || ordinal >= RUNTIME_STATE_MAX_TENSOR_EVENTS
        || layout->tensor_address == 0U
        || layout->row_size == 0U
        || layout->cell_capacity == 0U
        || layout->element_size == 0U
        || layout->embedding_width == 0U
        || layout->row_size > (~(__u32)0)
        || layout->cell_capacity > (~(__u32)0)
        || layout->element_size > (~(__u32)0)
        || layout->embedding_width > (~(__u32)0)
        || layout->element_size > RUNTIME_STATE_MAX_FRAGMENT_BYTES
        || layout->row_size
            != (__u64)(__u32)layout->element_size
                * (__u32)layout->embedding_width) {
        return false;
    }
    storage_size = (__u64)(__u32)layout->row_size
        * (__u32)layout->cell_capacity;
    return layout->tensor_address <= (~(__u64)0) - storage_size
        && bpf_map_update_elem(
            &runtime_state_retained_tensor_layouts,
            &ordinal,
            layout,
            BPF_ANY) == 0;
}

static __always_inline bool writer_boundary_mode(
    const struct runtime_state_target_config *config
) {
    return config != NULL
        && config->capture_armed == 1U
        && (config->capture_mode == RUNTIME_STATE_CAPTURE_WRITER_BOUNDARY
            || config->capture_mode
                == RUNTIME_STATE_CAPTURE_GATHER_WRITER_BOUNDARY);
}

static __always_inline bool legacy_writer_boundary_mode(
    const struct runtime_state_target_config *config
) {
    return config != NULL
        && config->capture_mode == RUNTIME_STATE_CAPTURE_WRITER_BOUNDARY;
}

static __always_inline bool gather_writer_boundary_mode(
    const struct runtime_state_target_config *config
) {
    return config != NULL
        && config->capture_mode
            == RUNTIME_STATE_CAPTURE_GATHER_WRITER_BOUNDARY;
}

static long observe_apply_token(
    __u32 index,
    void *data
) {
    struct apply_token_loop_context *loop = data;
    struct runtime_state_target_config *config;
    struct runtime_state_bpf_counters *counters;
    struct runtime_state_bpf_diagnostics *diagnostics;
    struct runtime_state_kernel_event *output;
    __u64 sequence_pointer = 0U;
    __u64 inner_vector = 0U;
    __u64 cell_indices = 0U;
    __u64 cell_indices_end = 0U;
    __s32 sequence_id = -1;
    __s32 position = -1;
    __s32 sequence_count = 0;
    __u32 cell_index = 0U;
    __u32 stream_ordinal;
    __u32 index_in_stream;
    __u32 stream_index = 0U;
    __u32 seq_key;
    __u32 *prior_stream;
    struct apply_call_state *apply_call;
    bool stream_found = false;

    if (loop == NULL) {
        return 1;
    }
    config = target_config(loop->pid_tgid);
    counters = global_counters();
    diagnostics = global_diagnostics();
    if (config == NULL || counters == NULL) {
        loop->failed = 1U;
        return 1;
    }
    index_in_stream = index;
#pragma unroll
    for (stream_ordinal = 0U;
         stream_ordinal < RUNTIME_STATE_MAX_APPLY_STREAMS;
         ++stream_ordinal) {
        __u32 stream_token_count;
        if (stream_ordinal >= loop->n_stream_in_call) {
            break;
        }
        inner_vector = loop->outer_idxs_begin + stream_ordinal * 24U;
        if (bpf_probe_read_user(&cell_indices, sizeof(cell_indices),
                (const void *)inner_vector) != 0
            || bpf_probe_read_user(&cell_indices_end, sizeof(cell_indices_end),
                (const void *)(inner_vector + 8U)) != 0) {
            record_read_failure(counters);
            if (diagnostics != NULL) {
                increment(&diagnostics->apply_token_read_failure_count);
            }
            loop->failed = 1U;
            return 1;
        }
        if (cell_indices == 0U || cell_indices_end <= cell_indices
            || (cell_indices_end - cell_indices) % sizeof(__u32) != 0U) {
            increment(&counters->unsupported_mutation_count);
            if (diagnostics != NULL) {
                increment(&diagnostics->apply_token_shape_invalid_count);
            }
            loop->failed = 1U;
            return 1;
        }
        stream_token_count = (__u32)(
            (cell_indices_end - cell_indices) / sizeof(__u32));
        if (index_in_stream < stream_token_count) {
            stream_found = true;
            break;
        }
        index_in_stream -= stream_token_count;
    }
    if (!stream_found) {
        increment(&counters->unsupported_mutation_count);
        if (diagnostics != NULL) {
            increment(&diagnostics->apply_token_shape_invalid_count);
        }
        loop->failed = 1U;
        return 1;
    }
    if (bpf_probe_read_user(&stream_index, sizeof(stream_index),
            (const void *)(loop->stream_begin
                + stream_ordinal * sizeof(__u32))) != 0
        || bpf_probe_read_user(&sequence_count, sizeof(sequence_count),
            (const void *)(loop->sequence_counts
                + index * sizeof(__s32))) != 0
        || bpf_probe_read_user(&sequence_pointer, sizeof(sequence_pointer),
            (const void *)(loop->sequences + index * sizeof(__u64))) != 0
        || bpf_probe_read_user(&sequence_id, sizeof(sequence_id),
            (const void *)sequence_pointer) != 0
        || bpf_probe_read_user(&position, sizeof(position),
            (const void *)(loop->positions + index * sizeof(__s32))) != 0
        || bpf_probe_read_user(&cell_index, sizeof(cell_index),
            (const void *)(cell_indices
                + index_in_stream * sizeof(__u32))) != 0) {
        record_read_failure(counters);
        if (diagnostics != NULL) {
            increment(&diagnostics->apply_token_read_failure_count);
        }
        loop->failed = 1U;
        return 1;
    }
    if (stream_index >= config->expected_n_stream
        || stream_index >= RUNTIME_STATE_MAX_APPLY_STREAMS
        || sequence_count != 1 || sequence_pointer == 0U
        || sequence_id < 0
        || (__u32)sequence_id >= config->expected_n_seq_max
        || position < 0) {
        increment(&counters->unsupported_mutation_count);
        if (diagnostics != NULL) {
            increment(&diagnostics->apply_token_shape_invalid_count);
        }
        loop->failed = 1U;
        return 1;
    }
    seq_key = (__u32)sequence_id;
    prior_stream = bpf_map_lookup_elem(&observed_sequence_stream, &seq_key);
    if (prior_stream != NULL && *prior_stream != 0U
        && *prior_stream != stream_index + 1U) {
        increment(&counters->mapping_mutation_count);
        emit_mutation(
            loop->pid_tgid,
            RUNTIME_STATE_MUTATION_SEQUENCE_STREAM_REMAP,
            counters);
    }
    {
        __u32 encoded_stream = stream_index + 1U;
        (void)bpf_map_update_elem(
            &observed_sequence_stream, &seq_key, &encoded_stream, BPF_ANY);
    }
    if (loop->capture_mode == RUNTIME_STATE_CAPTURE_WRITER_BOUNDARY
        || loop->capture_mode
            == RUNTIME_STATE_CAPTURE_GATHER_WRITER_BOUNDARY
        || loop->capture_mode == RUNTIME_STATE_CAPTURE_COMPACT_METADATA) {
        apply_call = bpf_map_lookup_elem(
            &apply_calls_by_thread, &loop->pid_tgid);
        if (apply_call == NULL
            || apply_call->apply_call_id != loop->apply_call_id) {
            increment(&counters->unmatched_call_count);
            if (diagnostics != NULL) {
                increment(&diagnostics->apply_state_conflict_count);
            }
            loop->failed = 1U;
            return 1;
        }
        stream_index &= RUNTIME_STATE_MAX_APPLY_STREAMS - 1U;
        if (index_in_stream == 0U) {
            apply_call->first_token_positions[stream_index] = (__u32)position;
            apply_call->first_cell_indices[stream_index] = cell_index;
            apply_call->sequence_ids[stream_index] = (__u32)sequence_id;
        } else if (
            apply_call->first_token_positions[stream_index]
                    + index_in_stream != (__u32)position
            || apply_call->first_cell_indices[stream_index]
                    + index_in_stream != cell_index
            || apply_call->sequence_ids[stream_index]
                    != (__u32)sequence_id) {
            increment(&counters->unsupported_mutation_count);
            if (diagnostics != NULL) {
                increment(&diagnostics->apply_token_shape_invalid_count);
            }
            loop->failed = 1U;
            return 1;
        }
        if (loop->capture_mode == RUNTIME_STATE_CAPTURE_COMPACT_METADATA) {
            return 0;
        }
    }
    output = global_event_scratch();
    if (output == NULL) {
        increment(&counters->unsupported_mutation_count);
        loop->failed = 1U;
        return 1;
    }
    __builtin_memset(output, 0, sizeof(*output));
    output->event_type = RUNTIME_STATE_KERNEL_TOKEN_CELL;
    output->payload.token.apply_call_id = loop->apply_call_id;
    output->payload.token.tgid = (__u32)(loop->pid_tgid >> 32U);
    output->payload.token.tid = (__u32)loop->pid_tgid;
    output->payload.token.monotonic_ns = bpf_ktime_get_ns();
    output->payload.token.seq_id = (__u32)sequence_id;
    output->payload.token.stream_index = stream_index;
    output->payload.token.token_position = (__u32)position;
    output->payload.token.cell_index = cell_index;
    if (loop->capture_mode == RUNTIME_STATE_CAPTURE_WRITER_BOUNDARY
        || (loop->capture_mode
                == RUNTIME_STATE_CAPTURE_GATHER_WRITER_BOUNDARY
            && config->capture_armed == 1U)) {
        const struct apply_token_plan_key key = {
            .apply_call_id = loop->apply_call_id,
            .token_ordinal = index,
        };
        if (bpf_map_update_elem(
                &apply_token_plan,
                &key,
                &output->payload.token,
                BPF_NOEXIST) != 0) {
            increment(&counters->unmatched_call_count);
            if (diagnostics != NULL) {
                increment(&diagnostics->apply_state_conflict_count);
            }
            loop->failed = 1U;
            return 1;
        }
        return 0;
    }
    (void)submit_event(output, counters, &counters->token_cell_event_count);
    return 0;
}

static __always_inline bool mark_stream_busy(
    struct apply_call_state *call,
    __u32 stream_index,
    __u64 *prior_version
) {
    __u64 *version;
    __u64 prior;
    __u64 busy;
    if (call == NULL || prior_version == NULL
        || stream_index >= RUNTIME_STATE_MAX_APPLY_STREAMS) {
        return false;
    }
    *prior_version = 0U;
    if ((call->stream_mask & (1U << stream_index)) == 0U) {
        return true;
    }
    version = bpf_map_lookup_elem(
        &runtime_state_stream_versions, &stream_index);
    if (version == NULL) {
        return false;
    }
    stream_index &= RUNTIME_STATE_MAX_APPLY_STREAMS - 1U;
    prior = *version;
    *prior_version = prior;
    call->prior_stream_versions[stream_index] = prior;
    if ((prior & RUNTIME_STATE_STREAM_VERSION_BUSY_FLAG) != 0U) {
        return false;
    }
    busy = RUNTIME_STATE_STREAM_VERSION_BUSY_FLAG | call->apply_call_id;
    return __sync_val_compare_and_swap(version, prior, busy) == prior;
}

static __always_inline void restore_stream_version(
    const struct apply_call_state *call,
    __u32 stream_index,
    __u64 prior_version
) {
    __u64 *version;
    __u64 busy;
    if (call == NULL || stream_index >= RUNTIME_STATE_MAX_APPLY_STREAMS
        || (call->stream_mask & (1U << stream_index)) == 0U) {
        return;
    }
    version = bpf_map_lookup_elem(
        &runtime_state_stream_versions, &stream_index);
    if (version == NULL) {
        return;
    }
    busy = RUNTIME_STATE_STREAM_VERSION_BUSY_FLAG | call->apply_call_id;
    (void)__sync_val_compare_and_swap(version, busy, prior_version);
}

static __always_inline bool mark_apply_streams_busy(
    struct apply_call_state *call
) {
    __u64 prior_0 = 0U;
    __u64 prior_1 = 0U;
    __u64 prior_2 = 0U;
    __u64 prior_3 = 0U;
    __u64 prior_4 = 0U;
    __u64 prior_5 = 0U;
    __u64 prior_6 = 0U;
    __u64 prior_7 = 0U;
    if (!mark_stream_busy(call, 0U, &prior_0)
        || !mark_stream_busy(call, 1U, &prior_1)
        || !mark_stream_busy(call, 2U, &prior_2)
        || !mark_stream_busy(call, 3U, &prior_3)
        || !mark_stream_busy(call, 4U, &prior_4)
        || !mark_stream_busy(call, 5U, &prior_5)
        || !mark_stream_busy(call, 6U, &prior_6)
        || !mark_stream_busy(call, 7U, &prior_7)) {
        restore_stream_version(call, 7U, prior_7);
        restore_stream_version(call, 6U, prior_6);
        restore_stream_version(call, 5U, prior_5);
        restore_stream_version(call, 4U, prior_4);
        restore_stream_version(call, 3U, prior_3);
        restore_stream_version(call, 2U, prior_2);
        restore_stream_version(call, 1U, prior_1);
        restore_stream_version(call, 0U, prior_0);
        return false;
    }
    return true;
}

static __always_inline void restore_apply_streams(
    const struct apply_call_state *call
) {
    if (call == NULL) {
        return;
    }
    restore_stream_version(call, 7U, call->prior_stream_versions[7]);
    restore_stream_version(call, 6U, call->prior_stream_versions[6]);
    restore_stream_version(call, 5U, call->prior_stream_versions[5]);
    restore_stream_version(call, 4U, call->prior_stream_versions[4]);
    restore_stream_version(call, 3U, call->prior_stream_versions[3]);
    restore_stream_version(call, 2U, call->prior_stream_versions[2]);
    restore_stream_version(call, 1U, call->prior_stream_versions[1]);
    restore_stream_version(call, 0U, call->prior_stream_versions[0]);
}

static __always_inline bool complete_stream_version(
    const struct apply_call_state *call,
    __u32 stream_index
) {
    __u64 *version;
    __u64 busy;
    if (call == NULL || stream_index >= RUNTIME_STATE_MAX_APPLY_STREAMS) {
        return false;
    }
    if ((call->stream_mask & (1U << stream_index)) == 0U) {
        return true;
    }
    version = bpf_map_lookup_elem(
        &runtime_state_stream_versions, &stream_index);
    if (version == NULL) {
        return false;
    }
    busy = RUNTIME_STATE_STREAM_VERSION_BUSY_FLAG | call->apply_call_id;
    return __sync_val_compare_and_swap(
        version, busy, call->apply_call_id) == busy;
}

static long finish_apply_token_plan(__u32 index, void *data) {
    struct apply_token_plan_loop_context *loop = data;
    struct runtime_state_bpf_counters *counters = global_counters();
    const struct apply_token_plan_key key = {
        .apply_call_id = loop->apply_call_id,
        .token_ordinal = index,
    };
    struct runtime_state_kernel_token_cell_event *token;
    struct runtime_state_kernel_event output = {0};
    if (counters == NULL || index >= loop->token_cell_count) {
        loop->failed = 1U;
        return 0;
    }
    token = bpf_map_lookup_elem(&apply_token_plan, &key);
    if (token == NULL) {
        loop->failed = 1U;
        return 0;
    }
    if (loop->emit != 0U) {
        output.event_type = RUNTIME_STATE_KERNEL_TOKEN_CELL;
        __builtin_memcpy(&output.payload.token, token, sizeof(*token));
        if (!submit_event(
                &output, counters, &counters->token_cell_event_count)) {
            loop->failed = 1U;
        }
    }
    if (bpf_map_delete_elem(&apply_token_plan, &key) != 0) {
        loop->failed = 1U;
    }
    return 0;
}

static __always_inline bool delete_apply_token_plan(
    __u64 pid_tgid,
    const struct apply_call_state *call,
    bool emit
) {
    struct apply_token_plan_loop_context loop;
    long processed;
    if (call == NULL || call->apply_call_id == 0U
        || call->token_cell_count == 0U
        || call->token_cell_count > RUNTIME_STATE_MAX_APPLY_TOKENS) {
        return false;
    }
    __builtin_memset(&loop, 0, sizeof(loop));
    loop.pid_tgid = pid_tgid;
    loop.apply_call_id = call->apply_call_id;
    loop.token_cell_count = call->token_cell_count;
    loop.emit = emit ? 1U : 0U;
    processed = bpf_loop(
        call->token_cell_count, finish_apply_token_plan, &loop, 0U);
    return processed == (long)call->token_cell_count && loop.failed == 0U;
}

SEC("uprobe")
int observe_process_ubatch_entry(struct pt_regs *ctx) {
    const __u64 pid_tgid = bpf_get_current_pid_tgid();
    struct runtime_state_target_config *config = target_config(pid_tgid);
    struct runtime_state_bpf_counters *counters;
    struct runtime_state_bpf_diagnostics *diagnostics;
    struct runtime_state_compute_call compute = {0};
    struct runtime_state_compute_call *active;
    __u32 ignored_depth = 1U;
    __u32 *prior_ignored_depth;
    if (!writer_boundary_mode(config)) {
        return 0;
    }
    counters = global_counters();
    diagnostics = global_diagnostics();
    if (counters == NULL) {
        return 0;
    }
    active = bpf_map_lookup_elem(&compute_calls_by_thread, &pid_tgid);
    if (active != NULL) {
        prior_ignored_depth = bpf_map_lookup_elem(
            &ignored_compute_returns_by_thread, &pid_tgid);
        if (prior_ignored_depth != NULL) {
            ignored_depth = *prior_ignored_depth + 1U;
        }
        (void)bpf_map_update_elem(
            &ignored_compute_returns_by_thread,
            &pid_tgid,
            &ignored_depth,
            BPF_ANY);
        if (diagnostics != NULL) {
            increment(&diagnostics->process_state_conflict_count);
        }
        return 0;
    }
    compute.context_address = PT_REGS_PARM1(ctx);
    compute.phase = RUNTIME_STATE_COMPUTE_ACTIVE;
    compute.payload_reserve_failure_baseline =
        counters->payload_ringbuf_reserve_fail_count;
    if (compute.context_address == 0U
        || bpf_map_update_elem(
            &compute_calls_by_thread,
            &pid_tgid,
            &compute,
            BPF_NOEXIST) != 0) {
        increment(&counters->unmatched_call_count);
        if (diagnostics != NULL) {
            increment(&diagnostics->process_state_conflict_count);
        }
        return 0;
    }
    increment(&counters->inflight_call_count);
    return 0;
}

SEC("uprobe")
int observe_apply_ubatch_entry(struct pt_regs *ctx) {
    const __u64 pid_tgid = bpf_get_current_pid_tgid();
    struct runtime_state_target_config *config = target_config(pid_tgid);
    struct runtime_state_bpf_counters *counters;
    struct runtime_state_bpf_diagnostics *diagnostics;
    struct runtime_state_compute_call *compute = NULL;
    struct apply_call_state *call;
    __u64 sinfo = PT_REGS_PARM2(ctx);
    __u64 ubatch = PT_REGS_PARM3(ctx);
    __u64 stream_begin = 0U;
    __u64 stream_end = 0U;
    __u64 outer_idxs_begin = 0U;
    __u64 outer_idxs_end = 0U;
    __u64 positions = 0U;
    __u64 sequence_counts = 0U;
    __u64 sequences = 0U;
    __u32 n_tokens = 0U;
    __u32 n_stream_in_call = 0U;
    __u32 stream_token_total = 0U;
    struct apply_token_loop_context token_loop = {0};
    long processed_tokens;
    if (config == NULL) {
        return 0;
    }
    counters = global_counters();
    diagnostics = global_diagnostics();
    if (counters == NULL) {
        return 0;
    }
    call = global_apply_call_scratch();
    if (call == NULL) {
        increment(&counters->unsupported_mutation_count);
        return 0;
    }
    __builtin_memset(call, 0, sizeof(*call));
    if (writer_boundary_mode(config)) {
        compute = bpf_map_lookup_elem(&compute_calls_by_thread, &pid_tgid);
        if (compute == NULL
            || compute->phase != RUNTIME_STATE_COMPUTE_ACTIVE) {
            return 0;
        }
    }
    if (bpf_map_lookup_elem(&apply_calls_by_thread, &pid_tgid) != NULL) {
        if (counters != NULL) {
            increment(&counters->unmatched_call_count);
            if (diagnostics != NULL) {
                increment(&diagnostics->apply_state_conflict_count);
            }
        }
        return 0;
    }
    call->apply_call_id = __sync_fetch_and_add(&counters->next_apply_call_id, 1U) + 1U;
    if (bpf_probe_read_user(&stream_begin, sizeof(stream_begin), (const void *)(sinfo + 8U)) != 0
        || bpf_probe_read_user(&stream_end, sizeof(stream_end), (const void *)(sinfo + 16U)) != 0
        || bpf_probe_read_user(&outer_idxs_begin, sizeof(outer_idxs_begin), (const void *)(sinfo + 32U)) != 0
        || bpf_probe_read_user(&outer_idxs_end, sizeof(outer_idxs_end), (const void *)(sinfo + 40U)) != 0
        || bpf_probe_read_user(&n_tokens, sizeof(n_tokens), (const void *)(ubatch + 4U)) != 0
        || bpf_probe_read_user(&positions, sizeof(positions), (const void *)(ubatch + 40U)) != 0
        || bpf_probe_read_user(&sequence_counts, sizeof(sequence_counts), (const void *)(ubatch + 48U)) != 0
        || bpf_probe_read_user(&sequences, sizeof(sequences), (const void *)(ubatch + 56U)) != 0) {
        record_read_failure(counters);
        if (diagnostics != NULL) {
            increment(&diagnostics->apply_header_read_failure_count);
        }
        return 0;
    }
    if (stream_begin == 0U || stream_end <= stream_begin
        || (stream_end - stream_begin) % sizeof(__u32) != 0U
        || outer_idxs_begin == 0U || outer_idxs_end < outer_idxs_begin) {
        increment(&counters->unsupported_mutation_count);
        if (diagnostics != NULL) {
            increment(&diagnostics->apply_slot_shape_invalid_count);
        }
        return 0;
    }
    n_stream_in_call = (__u32)((stream_end - stream_begin) / sizeof(__u32));
    if (n_stream_in_call == 0U) {
        increment(&counters->unsupported_mutation_count);
        if (diagnostics != NULL) {
            increment(&diagnostics->apply_batch_shape_invalid_count);
            increment(&diagnostics->apply_stream_count_zero_count);
        }
        return 0;
    }
    if (n_stream_in_call > config->expected_n_stream) {
        increment(&counters->unsupported_mutation_count);
        if (diagnostics != NULL) {
            increment(&diagnostics->apply_batch_shape_invalid_count);
            increment(&diagnostics->apply_stream_count_exceeds_count);
        }
        return 0;
    }
    if (outer_idxs_end != outer_idxs_begin + n_stream_in_call * 24U) {
        increment(&counters->unsupported_mutation_count);
        if (diagnostics != NULL) {
            increment(&diagnostics->apply_batch_shape_invalid_count);
            increment(&diagnostics->apply_index_vector_extent_invalid_count);
        }
        return 0;
    }
    if (n_tokens == 0U) {
        increment(&counters->unsupported_mutation_count);
        if (diagnostics != NULL) {
            increment(&diagnostics->apply_batch_shape_invalid_count);
            increment(&diagnostics->apply_token_count_zero_count);
        }
        return 0;
    }
    if (n_tokens > RUNTIME_STATE_MAX_APPLY_TOKENS) {
        increment(&counters->unsupported_mutation_count);
        if (diagnostics != NULL) {
            __u64 sample = ((__u64)n_tokens << 32U) | n_stream_in_call;
            increment(&diagnostics->apply_batch_shape_invalid_count);
            increment(&diagnostics->apply_token_count_exceeds_count);
            (void)__sync_val_compare_and_swap(
                &diagnostics->apply_token_count_exceeds_sample,
                0U,
                sample);
        }
        return 0;
    }
    {
        __u32 stream_ordinal;
#pragma unroll
        for (stream_ordinal = 0U;
             stream_ordinal < RUNTIME_STATE_MAX_APPLY_STREAMS;
             ++stream_ordinal) {
            __u64 cell_indices = 0U;
            __u64 cell_indices_end = 0U;
            __u64 inner_vector;
            __u32 stream_token_count;
            __u32 stream_index = 0U;
            if (stream_ordinal >= n_stream_in_call) {
                break;
            }
            inner_vector = outer_idxs_begin + stream_ordinal * 24U;
            if (bpf_probe_read_user(&cell_indices, sizeof(cell_indices),
                    (const void *)inner_vector) != 0
                || bpf_probe_read_user(
                    &cell_indices_end,
                    sizeof(cell_indices_end),
                    (const void *)(inner_vector + 8U)) != 0) {
                record_read_failure(counters);
                if (diagnostics != NULL) {
                    increment(&diagnostics->apply_token_read_failure_count);
                }
                return 0;
            }
            if (cell_indices == 0U || cell_indices_end <= cell_indices
                || (cell_indices_end - cell_indices) % sizeof(__u32) != 0U) {
                increment(&counters->unsupported_mutation_count);
                if (diagnostics != NULL) {
                    increment(&diagnostics->apply_batch_shape_invalid_count);
                    increment(&diagnostics->apply_token_shape_invalid_count);
                }
                return 0;
            }
            stream_token_count = (__u32)(
                (cell_indices_end - cell_indices) / sizeof(__u32));
            if (bpf_probe_read_user(
                    &stream_index,
                    sizeof(stream_index),
                    (const void *)(stream_begin
                        + stream_ordinal * sizeof(__u32))) != 0) {
                record_read_failure(counters);
                if (diagnostics != NULL) {
                    increment(&diagnostics->apply_token_read_failure_count);
                }
                return 0;
            }
            if (stream_index >= config->expected_n_stream
                || stream_index >= RUNTIME_STATE_MAX_APPLY_STREAMS
                || (call->stream_mask & (1U << stream_index)) != 0U) {
                increment(&counters->unsupported_mutation_count);
                if (diagnostics != NULL) {
                    increment(&diagnostics->apply_batch_shape_invalid_count);
                    increment(&diagnostics->apply_token_shape_invalid_count);
                }
                return 0;
            }
            stream_index &= RUNTIME_STATE_MAX_APPLY_STREAMS - 1U;
            call->stream_mask |= 1U << stream_index;
            call->stream_token_counts[stream_index] = stream_token_count;
            if (stream_token_count > n_tokens - stream_token_total) {
                increment(&counters->unsupported_mutation_count);
                if (diagnostics != NULL) {
                    increment(&diagnostics->apply_batch_shape_invalid_count);
                    increment(
                        &diagnostics->apply_token_stream_remainder_count);
                }
                return 0;
            }
            stream_token_total += stream_token_count;
        }
    }
    if (stream_token_total != n_tokens) {
        increment(&counters->unsupported_mutation_count);
        if (diagnostics != NULL) {
            increment(&diagnostics->apply_batch_shape_invalid_count);
            increment(&diagnostics->apply_token_stream_remainder_count);
        }
        return 0;
    }
    if (call->apply_call_id == 0U
        || (call->apply_call_id & RUNTIME_STATE_STREAM_VERSION_BUSY_FLAG) != 0U
        || call->stream_mask == 0U) {
        increment(&counters->unsupported_mutation_count);
        if (diagnostics != NULL) {
            increment(&diagnostics->apply_batch_shape_invalid_count);
        }
        return 0;
    }
    call->token_cell_count = n_tokens;
    if (bpf_map_update_elem(
            &apply_calls_by_thread, &pid_tgid, call, BPF_NOEXIST) != 0) {
        increment(&counters->unmatched_call_count);
        if (diagnostics != NULL) {
            increment(&diagnostics->apply_state_conflict_count);
        }
        return 0;
    }
    if (!mark_apply_streams_busy(call)) {
        increment(&counters->unsupported_mutation_count);
        if (diagnostics != NULL) {
            increment(&diagnostics->apply_state_conflict_count);
        }
        (void)bpf_map_delete_elem(&apply_calls_by_thread, &pid_tgid);
        return 0;
    }
    if (compute != NULL) {
        compute->apply_call_id = call->apply_call_id;
        compute->stream_mask = call->stream_mask;
        compute->token_cell_count = n_tokens;
        compute->phase = RUNTIME_STATE_COMPUTE_APPLY_PLANNED;
    } else {
        increment(&counters->inflight_call_count);
    }
    token_loop.pid_tgid = pid_tgid;
    token_loop.apply_call_id = call->apply_call_id;
    token_loop.stream_begin = stream_begin;
    token_loop.outer_idxs_begin = outer_idxs_begin;
    token_loop.positions = positions;
    token_loop.sequence_counts = sequence_counts;
    token_loop.sequences = sequences;
    token_loop.n_stream_in_call = n_stream_in_call;
    token_loop.capture_mode = config->capture_mode;
    processed_tokens = bpf_loop(
        n_tokens, observe_apply_token, &token_loop, 0U);
    if (processed_tokens != (long)n_tokens || token_loop.failed != 0U) {
        increment(&counters->unmatched_call_count);
        if (diagnostics != NULL) {
            increment(&diagnostics->apply_state_conflict_count);
        }
        restore_apply_streams(call);
        (void)delete_apply_token_plan(pid_tgid, call, false);
        (void)bpf_map_delete_elem(&apply_calls_by_thread, &pid_tgid);
        if (compute != NULL) {
            compute->apply_call_id = 0U;
            compute->stream_mask = 0U;
            compute->token_cell_count = 0U;
            compute->phase = RUNTIME_STATE_COMPUTE_ACTIVE;
        } else {
            (void)__sync_fetch_and_sub(&counters->inflight_call_count, 1U);
        }
    }
    return 0;
}

static __always_inline bool submit_compact_range(
    __u64 pid_tgid,
    const struct apply_call_state *call,
    __u32 stream_index,
    struct runtime_state_bpf_counters *counters
) {
    struct runtime_state_kernel_event *output;
    __u32 count;
    if (call == NULL || counters == NULL
        || stream_index >= RUNTIME_STATE_MAX_APPLY_STREAMS) {
        return false;
    }
    if ((call->stream_mask & (1U << stream_index)) == 0U) {
        return true;
    }
    count = call->stream_token_counts[stream_index];
    if (count == 0U || count > RUNTIME_STATE_MAX_APPLY_TOKENS) {
        return false;
    }
    output = global_event_scratch();
    if (output == NULL) {
        return false;
    }
    __builtin_memset(output, 0, sizeof(*output));
    output->event_type = RUNTIME_STATE_KERNEL_TOKEN_CELL_RANGE;
    output->payload.token_range.apply_call_id = call->apply_call_id;
    output->payload.token_range.tgid = (__u32)(pid_tgid >> 32U);
    output->payload.token_range.tid = (__u32)pid_tgid;
    output->payload.token_range.seq_id = call->sequence_ids[stream_index];
    output->payload.token_range.stream_index = stream_index;
    output->payload.token_range.first_token_position =
        call->first_token_positions[stream_index];
    output->payload.token_range.first_cell_index =
        call->first_cell_indices[stream_index];
    output->payload.token_range.token_count = count;
    if (!submit_event(output, counters, NULL)) {
        return false;
    }
    (void)__sync_fetch_and_add(&counters->token_cell_event_count, count);
    return true;
}

static __always_inline void complete_metadata_apply(
    __u64 pid_tgid,
    const struct apply_call_state *call,
    const struct runtime_state_target_config *config,
    struct runtime_state_bpf_counters *counters,
    struct runtime_state_bpf_diagnostics *diagnostics
) {
    struct runtime_state_kernel_event output = {0};
    bool ranges_complete = true;
    bool versions_complete;
    if (config != NULL
        && config->capture_mode == RUNTIME_STATE_CAPTURE_COMPACT_METADATA) {
        ranges_complete = submit_compact_range(pid_tgid, call, 0U, counters);
        ranges_complete &= submit_compact_range(pid_tgid, call, 1U, counters);
        ranges_complete &= submit_compact_range(pid_tgid, call, 2U, counters);
        ranges_complete &= submit_compact_range(pid_tgid, call, 3U, counters);
        ranges_complete &= submit_compact_range(pid_tgid, call, 4U, counters);
        ranges_complete &= submit_compact_range(pid_tgid, call, 5U, counters);
        ranges_complete &= submit_compact_range(pid_tgid, call, 6U, counters);
        ranges_complete &= submit_compact_range(pid_tgid, call, 7U, counters);
    }
    if (!ranges_complete) {
        increment(&counters->unmatched_call_count);
        if (diagnostics != NULL) {
            increment(&diagnostics->apply_state_conflict_count);
        }
        restore_apply_streams(call);
        (void)bpf_map_delete_elem(&apply_calls_by_thread, &pid_tgid);
        (void)__sync_fetch_and_sub(&counters->inflight_call_count, 1U);
        return;
    }
    versions_complete = complete_stream_version(call, 0U);
    versions_complete &= complete_stream_version(call, 1U);
    versions_complete &= complete_stream_version(call, 2U);
    versions_complete &= complete_stream_version(call, 3U);
    versions_complete &= complete_stream_version(call, 4U);
    versions_complete &= complete_stream_version(call, 5U);
    versions_complete &= complete_stream_version(call, 6U);
    versions_complete &= complete_stream_version(call, 7U);
    if (!versions_complete) {
        increment(&counters->unmatched_call_count);
        if (diagnostics != NULL) {
            increment(&diagnostics->apply_state_conflict_count);
        }
    }
    output.event_type = RUNTIME_STATE_KERNEL_APPLY_COMPLETE;
    output.payload.apply_complete.apply_call_id = call->apply_call_id;
    output.payload.apply_complete.tgid = (__u32)(pid_tgid >> 32U);
    output.payload.apply_complete.tid = (__u32)pid_tgid;
    output.payload.apply_complete.monotonic_ns = bpf_ktime_get_ns();
    (void)submit_event(
        &output, counters, &counters->apply_complete_event_count);
    if (bpf_map_delete_elem(&apply_calls_by_thread, &pid_tgid) != 0) {
        increment(&counters->unmatched_call_count);
        if (diagnostics != NULL) {
            increment(&diagnostics->apply_state_delete_failure_count);
        }
    } else {
        (void)__sync_fetch_and_sub(&counters->inflight_call_count, 1U);
    }
}

SEC("uretprobe")
int observe_apply_ubatch_return(struct pt_regs *ctx) {
    const __u64 pid_tgid = bpf_get_current_pid_tgid();
    struct runtime_state_target_config *config = target_config(pid_tgid);
    struct runtime_state_bpf_counters *counters;
    struct runtime_state_bpf_diagnostics *diagnostics;
    struct runtime_state_compute_call *compute;
    struct apply_call_state *call;
    if (config == NULL) {
        return 0;
    }
    counters = global_counters();
    diagnostics = global_diagnostics();
    if (counters == NULL) {
        return 0;
    }
    if (writer_boundary_mode(config)) {
        compute = bpf_map_lookup_elem(&compute_calls_by_thread, &pid_tgid);
        if (compute == NULL) {
            return 0;
        }
        call = bpf_map_lookup_elem(&apply_calls_by_thread, &pid_tgid);
        if (call == NULL
            || compute->phase != RUNTIME_STATE_COMPUTE_APPLY_PLANNED
            || compute->apply_call_id != call->apply_call_id) {
            if (diagnostics != NULL) {
                increment(&diagnostics->apply_return_unmatched_count);
            }
            return 0;
        }
        compute->phase = RUNTIME_STATE_COMPUTE_APPLY_RETURNED;
        (void)ctx;
        return 0;
    }
    call = bpf_map_lookup_elem(&apply_calls_by_thread, &pid_tgid);
    if (call == NULL) {
        increment(&counters->unmatched_call_count);
        if (diagnostics != NULL) {
            increment(&diagnostics->apply_return_unmatched_count);
        }
        return 0;
    }
    complete_metadata_apply(pid_tgid, call, config, counters, diagnostics);
    (void)ctx;
    return 0;
}

static __always_inline bool submit_state_unit_schedule_batch(
    struct flat_tensor_fragment_context *loop,
    struct runtime_state_bpf_counters *counters,
    struct runtime_state_kernel_state_unit_schedule_batch_event *batch
) {
    if (loop == NULL || counters == NULL || batch == NULL
        || loop->batch_record_count == 0U
        || loop->batch_record_count > RUNTIME_STATE_MAX_BATCH_FRAGMENTS
        || loop->batch_payload_size == 0U
        || loop->batch_payload_size > RUNTIME_STATE_MAX_FRAGMENT_BYTES) {
        return false;
    }
    batch->event_type = RUNTIME_STATE_KERNEL_STATE_UNIT_SCHEDULE_BATCH;
    batch->reserved = 0U;
    batch->record_count = loop->batch_record_count;
    batch->payload_size = loop->batch_payload_size;
    if (bpf_ringbuf_output(
            &runtime_state_payload_events,
            batch, sizeof(*batch), 0U) != 0) {
        increment(&counters->payload_ringbuf_reserve_fail_count);
        return false;
    }
    increment(&counters->state_unit_fragment_submit_count);
    loop->batch_record_count = 0U;
    loop->batch_payload_size = 0U;
    return true;
}

static __always_inline bool submit_state_unit_fragment(
    struct runtime_state_bpf_counters *counters,
    struct runtime_state_kernel_state_unit_fragment_event *fragment,
    __u32 payload_size
) {
    __u32 event_size;
    if (counters == NULL || fragment == NULL
        || payload_size == 0U
        || payload_size > RUNTIME_STATE_MAX_FRAGMENT_BYTES) {
        return false;
    }
    event_size = RUNTIME_STATE_KERNEL_STATE_UNIT_FRAGMENT_EVENT_PREFIX_SIZE
        + payload_size;
    if (event_size > sizeof(*fragment)
        || bpf_ringbuf_output(
            &runtime_state_payload_events,
            fragment, event_size, 0U) != 0) {
        increment(&counters->payload_ringbuf_reserve_fail_count);
        return false;
    }
    increment(&counters->state_unit_fragment_submit_count);
    return true;
}

static long copy_flat_tensor_fragment(__u32 flat_index, void *data) {
    struct flat_tensor_fragment_context *loop = data;
    struct runtime_state_bpf_counters *counters = global_counters();
    struct runtime_state_kernel_tensor_layout_event *layout;
    struct runtime_state_kernel_state_unit_schedule_batch_event *destination;
    struct runtime_state_kernel_state_unit_fragment_event *single;
    __u64 storage_size;
    __u64 source_offset;
    __u64 source_address;
    __u64 cell_position;
    __u32 tensor_ordinal;
    __u32 fragment_in_tensor;
    __u32 layer;
    __u32 unit_size;
    __u32 cell_offset;
    __u32 cell_count;
    __u32 units_per_fragment;
    __u32 dimension = 0U;
    __u32 chunk_index;
    __u32 payload_size;
    __u32 record_index;
    __u32 payload_slot_offset;
    __u32 scratch_key = 0U;
    volatile __u32 computed_payload_size;
    if (counters == NULL || loop == NULL || loop->failed != 0U
        || loop->expected_n_stream == 0U
        || loop->expected_n_stream > RUNTIME_STATE_MAX_APPLY_STREAMS
        || loop->expected_n_layer == 0U
        || loop->chunks_per_dimension == 0U
        || loop->fragments_per_tensor == 0U
        || loop->stream_index >= loop->expected_n_stream
        || loop->stream_index >= RUNTIME_STATE_MAX_APPLY_STREAMS
        || loop->kind_offset > 1U) {
        return 1;
    }
    layer = flat_index / loop->fragments_per_tensor;
    fragment_in_tensor = flat_index % loop->fragments_per_tensor;
    if (layer >= loop->expected_n_layer) {
        loop->failed = 1U;
        return 1;
    }
    tensor_ordinal = layer * (2U * loop->expected_n_stream);
    tensor_ordinal += loop->stream_index * 2U + loop->kind_offset;
    if (tensor_ordinal >= RUNTIME_STATE_MAX_TENSOR_EVENTS) {
        loop->failed = 1U;
        return 1;
    }
    layout = bpf_map_lookup_elem(
        &runtime_state_retained_tensor_layouts, &tensor_ordinal);
    if (layout == NULL
        || layout->layer != layer
        || layout->stream_index != loop->stream_index
        || layout->kind != loop->tensor_kind
        || layout->tensor_address == 0U
        || layout->row_size != loop->row_size
        || layout->cell_capacity != loop->cell_capacity
        || layout->element_size != loop->element_size
        || layout->embedding_width != loop->embedding_width) {
        loop->failed = 1U;
        return 1;
    }
    storage_size = (__u64)loop->row_size * loop->cell_capacity;
    if (layout->tensor_address > (~(__u64)0) - storage_size) {
        loop->failed = 1U;
        return 1;
    }
    if (loop->transposed_value != 0U) {
        dimension = fragment_in_tensor / loop->chunks_per_dimension;
        chunk_index = fragment_in_tensor % loop->chunks_per_dimension;
        unit_size = loop->element_size;
        if (dimension >= loop->embedding_width) {
            loop->failed = 1U;
            return 1;
        }
    } else {
        chunk_index = fragment_in_tensor;
        unit_size = loop->row_size;
    }
    if (unit_size == 0U || unit_size > RUNTIME_STATE_MAX_FRAGMENT_BYTES) {
        loop->failed = 1U;
        return 1;
    }
    units_per_fragment = RUNTIME_STATE_MAX_FRAGMENT_BYTES / unit_size;
    if (units_per_fragment == 0U) {
        loop->failed = 1U;
        return 1;
    }
    cell_offset = chunk_index * units_per_fragment;
    if (cell_offset >= loop->token_count) {
        loop->failed = 1U;
        return 1;
    }
    cell_count = loop->token_count - cell_offset;
    if (cell_count > units_per_fragment) {
        cell_count = units_per_fragment;
    }
    computed_payload_size = (__u32)(cell_count * unit_size);
    payload_size = computed_payload_size;
    if (payload_size == 0U
        || payload_size > RUNTIME_STATE_MAX_FRAGMENT_BYTES) {
        loop->failed = 1U;
        return 1;
    }
    cell_position = (__u64)loop->first_cell_index + cell_offset;
    if (cell_position >= loop->cell_capacity
        || cell_count > loop->cell_capacity - cell_position) {
        loop->failed = 1U;
        return 1;
    }
    if (loop->transposed_value != 0U) {
        source_offset = (__u64)dimension * loop->element_size;
        source_offset *= loop->cell_capacity;
        source_offset += cell_position * loop->element_size;
    } else {
        source_offset = cell_position * loop->row_size;
    }
    if (source_offset >= storage_size
        || payload_size > storage_size - source_offset
        || layout->tensor_address > (~(__u64)0) - source_offset
        || loop->fragment_base > (~(__u32)0) - flat_index) {
        loop->failed = 1U;
        return 1;
    }
    source_address = layout->tensor_address + source_offset;
    destination = bpf_map_lookup_elem(
        &runtime_state_payload_scratch, &scratch_key);
    if (destination == NULL) {
        increment(&counters->payload_ringbuf_reserve_fail_count);
        loop->failed = 1U;
        return 1;
    }
    if (loop->transposed_value == 0U
        || payload_size > RUNTIME_STATE_MAX_BATCH_FRAGMENT_BYTES) {
        if (loop->batch_record_count != 0U
            && !submit_state_unit_schedule_batch(
                loop, counters, destination)) {
            loop->failed = 1U;
            return 1;
        }
        single = bpf_map_lookup_elem(
            &runtime_state_fragment_scratch, &scratch_key);
        if (single == NULL) {
            increment(&counters->payload_ringbuf_reserve_fail_count);
            loop->failed = 1U;
            return 1;
        }
        single->event_type = RUNTIME_STATE_KERNEL_STATE_UNIT_FRAGMENT;
        single->reserved = 0U;
        single->payload.apply_call_id = loop->apply_call_id;
        single->payload.fragment_index = loop->fragment_base + flat_index;
        single->payload.seq_id = loop->seq_id;
        single->payload.stream_index = loop->stream_index;
        single->payload.tensor_kind = loop->tensor_kind;
        single->payload.token_position =
            loop->first_token_position + cell_offset;
        single->payload.cell_index = loop->first_cell_index + cell_offset;
        single->payload.layer = layer;
        single->payload.payload_size = payload_size;
        single->payload.tensor_byte_offset = source_offset;
        single->payload.writer_monotonic_ns = loop->writer_monotonic_ns;
        single->payload.span_cell_count = cell_count;
        payload_size = computed_payload_size;
        if (payload_size == 0U
            || payload_size > RUNTIME_STATE_MAX_FRAGMENT_BYTES
            || bpf_probe_read_user(single->payload.payload,
                payload_size, (const void *)source_address) != 0) {
            increment(&counters->payload_user_memory_read_failure_count);
            loop->failed = 1U;
            return 1;
        }
        if (!submit_state_unit_fragment(counters, single, payload_size)) {
            loop->failed = 1U;
            return 1;
        }
        return 0;
    }
    if (runtime_state_kernel_schedule_batch_can_append(
            loop->batch_record_count,
            loop->batch_payload_size,
            payload_size) == 0U) {
        if (loop->batch_record_count == 0U
            || !submit_state_unit_schedule_batch(
                loop, counters, destination)
            || runtime_state_kernel_schedule_batch_can_append(
                0U, 0U, payload_size) == 0U) {
            loop->failed = 1U;
            return 1;
        }
    }
    record_index = loop->batch_record_count;
    if (record_index >= RUNTIME_STATE_MAX_BATCH_FRAGMENTS) {
        loop->failed = 1U;
        return 1;
    }
    record_index &= RUNTIME_STATE_MAX_BATCH_FRAGMENTS - 1U;
    payload_slot_offset =
        record_index * RUNTIME_STATE_MAX_BATCH_FRAGMENT_BYTES;
    if (payload_slot_offset
            > RUNTIME_STATE_MAX_FRAGMENT_BYTES
                - RUNTIME_STATE_MAX_BATCH_FRAGMENT_BYTES) {
        loop->failed = 1U;
        return 1;
    }
    if (record_index == 0U) {
        destination->apply_call_id = loop->apply_call_id;
        destination->first_token_position = loop->first_token_position;
        destination->first_cell_index = loop->first_cell_index;
        destination->writer_monotonic_ns = loop->writer_monotonic_ns;
        destination->first_fragment_index =
            loop->fragment_base + flat_index;
        destination->first_flat_index = flat_index;
        destination->seq_id = loop->seq_id;
        destination->stream_index = loop->stream_index;
        destination->tensor_kind = loop->tensor_kind;
        destination->fragments_per_tensor = loop->fragments_per_tensor;
        destination->chunks_per_dimension = loop->chunks_per_dimension;
        destination->token_count = loop->token_count;
        destination->row_size = loop->row_size;
        destination->cell_capacity = loop->cell_capacity;
        destination->element_size = loop->element_size;
        destination->embedding_width = loop->embedding_width;
        destination->transposed_value = loop->transposed_value;
        destination->schedule_reserved = 0U;
    }
    payload_size = computed_payload_size;
    if (payload_size == 0U
        || payload_size > RUNTIME_STATE_MAX_BATCH_FRAGMENT_BYTES) {
        loop->failed = 1U;
        return 1;
    }
    if (bpf_probe_read_user(destination->payload + payload_slot_offset,
            payload_size, (const void *)source_address) != 0) {
        increment(&counters->payload_user_memory_read_failure_count);
        loop->failed = 1U;
        return 1;
    }
    loop->batch_record_count += 1U;
    loop->batch_payload_size += payload_size;
    return 0;
}

static __noinline bool run_flat_fragment_schedule(
    struct state_unit_copy_loop_context *copy,
    const struct apply_call_state *call,
    __u32 stream_index,
    __u32 tensor_kind
) {
    struct runtime_state_kernel_tensor_layout_event *layout;
    struct flat_tensor_fragment_context fragments = {0};
    __u32 kind_offset;
    __u32 first_ordinal;
    __u32 unit_size;
    __u32 units_per_fragment;
    __u64 storage_size;
    __u64 fragments_per_tensor;
    __u64 total_fragments;
    __u64 schedule_copy_byte_count;
    long processed;
    if (copy == NULL || call == NULL || copy->failed != 0U
        || call->apply_call_id != copy->apply_call_id
        || stream_index >= RUNTIME_STATE_MAX_APPLY_STREAMS) {
        return false;
    }
    if (stream_index >= copy->expected_n_stream) {
        copy->failed = 1U;
        return false;
    }
    stream_index &= RUNTIME_STATE_MAX_APPLY_STREAMS - 1U;
    if (tensor_kind == RUNTIME_STATE_TENSOR_KIND_K) {
        kind_offset = 0U;
    } else if (tensor_kind == RUNTIME_STATE_TENSOR_KIND_V) {
        kind_offset = 1U;
    } else {
        copy->failed = 1U;
        return false;
    }
    first_ordinal = stream_index * 2U + kind_offset;
    if (first_ordinal >= RUNTIME_STATE_MAX_TENSOR_EVENTS) {
        copy->failed = 1U;
        return false;
    }
    layout = bpf_map_lookup_elem(
        &runtime_state_retained_tensor_layouts, &first_ordinal);
    if (layout == NULL) {
        copy->failed = 1U;
        return false;
    }
    if (layout->layer != 0U
        || layout->stream_index != stream_index
        || layout->kind != tensor_kind
        || layout->tensor_address == 0U
        || layout->row_size == 0U
        || layout->cell_capacity == 0U
        || layout->element_size == 0U
        || layout->embedding_width == 0U
        || layout->row_size > (~(__u32)0)
        || layout->cell_capacity > (~(__u32)0)
        || layout->element_size > (~(__u32)0)
        || layout->embedding_width > (~(__u32)0)) {
        copy->failed = 1U;
        return false;
    }
    if (call->first_cell_indices[stream_index] > (~(__u32)0)
        || call->first_token_positions[stream_index]
            > (~(__u64)0)
                - call->stream_token_counts[stream_index]) {
        copy->failed = 1U;
        return false;
    }
    storage_size = (__u64)(__u32)layout->row_size
        * (__u32)layout->cell_capacity;
    if (layout->tensor_address > (~(__u64)0) - storage_size) {
        copy->failed = 1U;
        return false;
    }
    fragments.apply_call_id = call->apply_call_id;
    fragments.writer_monotonic_ns = copy->writer_monotonic_ns;
    fragments.expected_n_stream = copy->expected_n_stream;
    fragments.expected_n_layer = copy->expected_n_layer;
    fragments.row_size = (__u32)layout->row_size;
    fragments.cell_capacity = (__u32)layout->cell_capacity;
    fragments.element_size = (__u32)layout->element_size;
    fragments.embedding_width = (__u32)layout->embedding_width;
    fragments.first_token_position =
        call->first_token_positions[stream_index];
    fragments.first_cell_index = (__u32)
        call->first_cell_indices[stream_index];
    fragments.fragment_base = copy->fragment_index;
    fragments.seq_id = call->sequence_ids[stream_index];
    fragments.stream_index = stream_index;
    fragments.tensor_kind = tensor_kind;
    fragments.kind_offset = kind_offset;
    fragments.token_count = call->stream_token_counts[stream_index];
    fragments.transposed_value = copy->v_trans != 0U
            && tensor_kind == RUNTIME_STATE_TENSOR_KIND_V
        ? 1U : 0U;
    unit_size = fragments.transposed_value != 0U
        ? layout->element_size : layout->row_size;
    if (fragments.token_count == 0U
        || fragments.token_count > RUNTIME_STATE_MAX_APPLY_TOKENS
        || fragments.first_cell_index >= layout->cell_capacity
        || fragments.token_count
            > layout->cell_capacity - fragments.first_cell_index
        || layout->row_size
            != (__u64)(__u32)layout->element_size
                * (__u32)layout->embedding_width
        || unit_size == 0U
        || unit_size > RUNTIME_STATE_MAX_FRAGMENT_BYTES) {
        copy->failed = 1U;
        return false;
    }
    units_per_fragment = RUNTIME_STATE_MAX_FRAGMENT_BYTES / unit_size;
    if (units_per_fragment == 0U) {
        copy->failed = 1U;
        return false;
    }
    fragments.chunks_per_dimension = (__u32)(
        (fragments.token_count + units_per_fragment - 1U)
            / units_per_fragment);
    fragments_per_tensor = fragments.chunks_per_dimension;
    if (fragments.transposed_value != 0U) {
        fragments_per_tensor = (__u64)fragments.chunks_per_dimension
            * fragments.embedding_width;
    }
    total_fragments = fragments_per_tensor * copy->expected_n_layer;
    if (fragments_per_tensor == 0U
        || fragments_per_tensor > (~(__u32)0)
        || total_fragments == 0U || total_fragments > (~(__u32)0)
        || copy->fragment_index
            > (~(__u32)0) - (__u32)total_fragments) {
        copy->failed = 1U;
        return false;
    }
    schedule_copy_byte_count = (__u64)fragments.token_count
        * fragments.row_size * copy->expected_n_layer;
    if (schedule_copy_byte_count == 0U
        || copy->copy_byte_count
            > (~(__u64)0) - schedule_copy_byte_count) {
        copy->failed = 1U;
        return false;
    }
    fragments.fragments_per_tensor = (__u32)fragments_per_tensor;
    processed = bpf_loop(
        (__u32)total_fragments,
        copy_flat_tensor_fragment,
        &fragments,
        0U);
    if (processed != (long)total_fragments || fragments.failed != 0U) {
        copy->failed = 1U;
        return false;
    }
    if (fragments.batch_record_count != 0U) {
        __u32 scratch_key = 0U;
        struct runtime_state_bpf_counters *counters = global_counters();
        struct runtime_state_kernel_state_unit_schedule_batch_event *batch
            = bpf_map_lookup_elem(
                &runtime_state_payload_scratch, &scratch_key);
        if (counters == NULL || batch == NULL
            || !submit_state_unit_schedule_batch(
                &fragments, counters, batch)) {
            copy->failed = 1U;
            return false;
        }
    }
    copy->fragment_index += (__u32)total_fragments;
    copy->copy_byte_count += schedule_copy_byte_count;
    return true;
}

static long publish_read_plan_for_layer(__u32 layer, void *data) {
    struct read_plan_schedule_context *loop = data;
    struct runtime_state_bpf_counters *counters = global_counters();
    struct runtime_state_kernel_tensor_layout_event *layout;
    struct runtime_state_kernel_state_unit_read_plan_event *destination;
    __u32 tensor_ordinal;
    __u64 ordinal;
    __u64 storage_size;
    __u64 layer_fragment_offset;
    if (loop == NULL || counters == NULL || loop->failed != 0U
        || layer >= loop->expected_n_layer
        || loop->expected_n_stream == 0U
        || loop->expected_n_stream > RUNTIME_STATE_MAX_APPLY_STREAMS) {
        if (loop != NULL) {
            loop->failed = 1U;
        }
        return 1;
    }
    ordinal = (__u64)layer * (2U * loop->expected_n_stream)
        + (__u64)loop->stream_index * 2U + loop->kind_offset;
    if (ordinal >= RUNTIME_STATE_MAX_TENSOR_EVENTS) {
        loop->failed = 1U;
        return 1;
    }
    tensor_ordinal = (__u32)ordinal;
    layout = bpf_map_lookup_elem(
        &runtime_state_retained_tensor_layouts, &tensor_ordinal);
    if (layout == NULL || layout->layer != layer
        || layout->stream_index != loop->stream_index
        || layout->kind != loop->tensor_kind
        || layout->tensor_address == 0U
        || layout->row_size != loop->row_size
        || layout->cell_capacity != loop->cell_capacity
        || layout->element_size != loop->element_size
        || layout->embedding_width != loop->embedding_width) {
        loop->failed = 1U;
        return 1;
    }
    storage_size = (__u64)loop->row_size * loop->cell_capacity;
    layer_fragment_offset = (__u64)layer
        * loop->fragments_per_tensor;
    if (storage_size == 0U
        || layout->tensor_address > (~(__u64)0) - storage_size
        || layer_fragment_offset > (~(__u32)0)
        || loop->fragment_base
            > (~(__u32)0) - (__u32)layer_fragment_offset
        || loop->fragment_base + (__u32)layer_fragment_offset
            > (~(__u32)0) - loop->fragments_per_tensor) {
        loop->failed = 1U;
        return 1;
    }
    destination = bpf_ringbuf_reserve(
        &runtime_state_payload_events, sizeof(*destination), 0U);
    if (destination == NULL) {
        increment(&counters->payload_ringbuf_reserve_fail_count);
        loop->failed = 1U;
        return 1;
    }
    destination->event_type = RUNTIME_STATE_KERNEL_STATE_UNIT_READ_PLAN;
    destination->reserved = 0U;
    destination->apply_call_id = loop->apply_call_id;
    destination->tensor_address = layout->tensor_address;
    destination->first_token_position = loop->first_token_position;
    destination->first_cell_index = loop->first_cell_index;
    destination->writer_monotonic_ns = loop->writer_monotonic_ns;
    destination->copy_byte_count = loop->per_layer_copy_byte_count;
    destination->first_fragment_index = loop->fragment_base
        + (__u32)layer_fragment_offset;
    destination->fragment_count = loop->fragments_per_tensor;
    destination->seq_id = loop->seq_id;
    destination->stream_index = loop->stream_index;
    destination->tensor_kind = loop->tensor_kind;
    destination->layer = layer;
    destination->expected_n_layer = loop->expected_n_layer;
    destination->token_count = loop->token_count;
    destination->row_size = loop->row_size;
    destination->cell_capacity = loop->cell_capacity;
    destination->element_size = loop->element_size;
    destination->embedding_width = loop->embedding_width;
    destination->chunks_per_dimension = loop->chunks_per_dimension;
    destination->fragments_per_tensor = loop->fragments_per_tensor;
    destination->transposed_value = loop->transposed_value;
    destination->plan_reserved = 0U;
    bpf_ringbuf_submit(destination, 0U);
    increment(&counters->state_unit_fragment_submit_count);
    return 0;
}

static __noinline bool run_read_plan_schedule(
    struct state_unit_copy_loop_context *copy,
    const struct apply_call_state *call,
    __u32 stream_index,
    __u32 tensor_kind
) {
    struct runtime_state_kernel_tensor_layout_event *layout;
    struct read_plan_schedule_context plan = {0};
    __u32 kind_offset;
    __u32 first_ordinal;
    __u32 unit_size;
    __u32 units_per_fragment;
    __u64 fragments_per_tensor;
    __u64 total_fragments;
    __u64 schedule_copy_byte_count;
    long processed;
    if (copy == NULL || call == NULL || copy->failed != 0U
        || call->apply_call_id != copy->apply_call_id
        || stream_index >= copy->expected_n_stream
        || stream_index >= RUNTIME_STATE_MAX_APPLY_STREAMS) {
        return false;
    }
    stream_index &= RUNTIME_STATE_MAX_APPLY_STREAMS - 1U;
    if (tensor_kind == RUNTIME_STATE_TENSOR_KIND_K) {
        kind_offset = 0U;
    } else if (tensor_kind == RUNTIME_STATE_TENSOR_KIND_V) {
        kind_offset = 1U;
    } else {
        copy->failed = 1U;
        return false;
    }
    first_ordinal = stream_index * 2U + kind_offset;
    if (first_ordinal >= RUNTIME_STATE_MAX_TENSOR_EVENTS) {
        copy->failed = 1U;
        return false;
    }
    layout = bpf_map_lookup_elem(
        &runtime_state_retained_tensor_layouts, &first_ordinal);
    if (layout == NULL || layout->layer != 0U
        || layout->stream_index != stream_index
        || layout->kind != tensor_kind
        || layout->tensor_address == 0U
        || layout->row_size == 0U
        || layout->cell_capacity == 0U
        || layout->element_size == 0U
        || layout->embedding_width == 0U
        || layout->row_size > (~(__u32)0)
        || layout->cell_capacity > (~(__u32)0)
        || layout->element_size > (~(__u32)0)
        || layout->embedding_width > (~(__u32)0)
        || call->first_cell_indices[stream_index] > (~(__u32)0)
        || call->first_token_positions[stream_index]
            > (~(__u64)0) - call->stream_token_counts[stream_index]) {
        copy->failed = 1U;
        return false;
    }
    plan.apply_call_id = call->apply_call_id;
    plan.writer_monotonic_ns = copy->writer_monotonic_ns;
    plan.first_token_position = call->first_token_positions[stream_index];
    plan.first_cell_index = call->first_cell_indices[stream_index];
    plan.expected_n_stream = copy->expected_n_stream;
    plan.expected_n_layer = copy->expected_n_layer;
    plan.row_size = (__u32)layout->row_size;
    plan.cell_capacity = (__u32)layout->cell_capacity;
    plan.element_size = (__u32)layout->element_size;
    plan.embedding_width = (__u32)layout->embedding_width;
    plan.fragment_base = copy->fragment_index;
    plan.seq_id = call->sequence_ids[stream_index];
    plan.stream_index = stream_index;
    plan.tensor_kind = tensor_kind;
    plan.kind_offset = kind_offset;
    plan.token_count = call->stream_token_counts[stream_index];
    plan.transposed_value = copy->v_trans != 0U
            && tensor_kind == RUNTIME_STATE_TENSOR_KIND_V
        ? 1U : 0U;
    unit_size = plan.transposed_value != 0U
        ? plan.element_size : plan.row_size;
    if (plan.token_count == 0U
        || plan.token_count > RUNTIME_STATE_MAX_APPLY_TOKENS
        || plan.first_cell_index >= plan.cell_capacity
        || plan.token_count > plan.cell_capacity - plan.first_cell_index
        || plan.row_size
            != plan.element_size * plan.embedding_width
        || unit_size == 0U
        || unit_size > RUNTIME_STATE_MAX_FRAGMENT_BYTES) {
        copy->failed = 1U;
        return false;
    }
    units_per_fragment = RUNTIME_STATE_MAX_FRAGMENT_BYTES / unit_size;
    if (units_per_fragment == 0U) {
        copy->failed = 1U;
        return false;
    }
    plan.chunks_per_dimension =
        (plan.token_count + units_per_fragment - 1U)
            / units_per_fragment;
    fragments_per_tensor = plan.chunks_per_dimension;
    if (plan.transposed_value != 0U) {
        fragments_per_tensor *= plan.embedding_width;
    }
    total_fragments = fragments_per_tensor * copy->expected_n_layer;
    schedule_copy_byte_count = (__u64)plan.token_count
        * plan.row_size * copy->expected_n_layer;
    if (fragments_per_tensor == 0U
        || fragments_per_tensor > (~(__u32)0)
        || total_fragments == 0U || total_fragments > (~(__u32)0)
        || copy->fragment_index > (~(__u32)0) - total_fragments
        || copy->plan_count > (~(__u32)0) - copy->expected_n_layer
        || schedule_copy_byte_count == 0U
        || copy->copy_byte_count
            > (~(__u64)0) - schedule_copy_byte_count) {
        copy->failed = 1U;
        return false;
    }
    plan.fragments_per_tensor = (__u32)fragments_per_tensor;
    plan.per_layer_copy_byte_count = (__u64)plan.token_count
        * plan.row_size;
    processed = bpf_loop(
        copy->expected_n_layer,
        publish_read_plan_for_layer,
        &plan,
        0U);
    if (processed != (long)copy->expected_n_layer
        || plan.failed != 0U) {
        copy->failed = 1U;
        return false;
    }
    copy->fragment_index += (__u32)total_fragments;
    copy->plan_count += copy->expected_n_layer;
    copy->copy_byte_count += schedule_copy_byte_count;
    return true;
}

static __always_inline bool plan_writer_state_unit(
    const struct apply_call_state *call,
    struct runtime_state_bpf_counters *counters,
    struct state_unit_copy_loop_context *copy
) {
    const __u32 zero = 0U;
    struct runtime_state_kernel_topology_event *topology;
    if (call == NULL || counters == NULL || copy == NULL) {
        return false;
    }
    topology = bpf_map_lookup_elem(&runtime_state_retained_topology, &zero);
    if (topology == NULL || topology->n_stream == 0U
        || topology->n_layer == 0U
        || topology->n_stream > RUNTIME_STATE_MAX_APPLY_STREAMS
        || topology->n_layer > RUNTIME_STATE_MAX_TENSOR_EVENTS
                / (2U * topology->n_stream)) {
        return false;
    }
    copy->apply_call_id = call->apply_call_id;
    copy->writer_monotonic_ns = bpf_ktime_get_ns();
    copy->expected_n_stream = topology->n_stream;
    copy->expected_n_layer = topology->n_layer;
    copy->v_trans = topology->v_trans;
    increment(&counters->active_emitter_count);
    if (((call->stream_mask & 1U) != 0U
            && (!run_read_plan_schedule(copy, call, 0U, RUNTIME_STATE_TENSOR_KIND_K)
                || !run_read_plan_schedule(copy, call, 0U, RUNTIME_STATE_TENSOR_KIND_V)))
        || ((call->stream_mask & 2U) != 0U
            && (!run_read_plan_schedule(copy, call, 1U, RUNTIME_STATE_TENSOR_KIND_K)
                || !run_read_plan_schedule(copy, call, 1U, RUNTIME_STATE_TENSOR_KIND_V)))
        || ((call->stream_mask & 4U) != 0U
            && (!run_read_plan_schedule(copy, call, 2U, RUNTIME_STATE_TENSOR_KIND_K)
                || !run_read_plan_schedule(copy, call, 2U, RUNTIME_STATE_TENSOR_KIND_V)))
        || ((call->stream_mask & 8U) != 0U
            && (!run_read_plan_schedule(copy, call, 3U, RUNTIME_STATE_TENSOR_KIND_K)
                || !run_read_plan_schedule(copy, call, 3U, RUNTIME_STATE_TENSOR_KIND_V)))
        || ((call->stream_mask & 16U) != 0U
            && (!run_read_plan_schedule(copy, call, 4U, RUNTIME_STATE_TENSOR_KIND_K)
                || !run_read_plan_schedule(copy, call, 4U, RUNTIME_STATE_TENSOR_KIND_V)))
        || ((call->stream_mask & 32U) != 0U
            && (!run_read_plan_schedule(copy, call, 5U, RUNTIME_STATE_TENSOR_KIND_K)
                || !run_read_plan_schedule(copy, call, 5U, RUNTIME_STATE_TENSOR_KIND_V)))
        || ((call->stream_mask & 64U) != 0U
            && (!run_read_plan_schedule(copy, call, 6U, RUNTIME_STATE_TENSOR_KIND_K)
                || !run_read_plan_schedule(copy, call, 6U, RUNTIME_STATE_TENSOR_KIND_V)))
        || ((call->stream_mask & 128U) != 0U
            && (!run_read_plan_schedule(copy, call, 7U, RUNTIME_STATE_TENSOR_KIND_K)
                || !run_read_plan_schedule(copy, call, 7U, RUNTIME_STATE_TENSOR_KIND_V)))
        || copy->failed != 0U || copy->plan_count == 0U
        || copy->fragment_index == 0U || copy->copy_byte_count == 0U) {
        (void)__sync_fetch_and_sub(&counters->active_emitter_count, 1U);
        return false;
    }
    (void)__sync_fetch_and_sub(&counters->active_emitter_count, 1U);
    return true;
}

static __always_inline bool submit_state_unit_apply_ready(
    const struct apply_call_state *call,
    const struct runtime_state_compute_call *compute,
    const struct state_unit_copy_loop_context *copy,
    struct runtime_state_bpf_counters *counters
) {
    struct runtime_state_kernel_state_unit_apply_ready_event *destination;
    if (call == NULL || compute == NULL || copy == NULL || counters == NULL
        || copy->plan_count == 0U || copy->fragment_index == 0U
        || copy->copy_byte_count == 0U
        || compute->payload_reserve_failure_baseline
            != counters->payload_ringbuf_reserve_fail_count) {
        return false;
    }
    destination = bpf_ringbuf_reserve(
        &runtime_state_payload_events, sizeof(*destination), 0U);
    if (destination == NULL) {
        increment(&counters->payload_ringbuf_reserve_fail_count);
        return false;
    }
    destination->event_type = RUNTIME_STATE_KERNEL_STATE_UNIT_APPLY_READY;
    destination->reserved = 0U;
    destination->payload.apply_call_id = call->apply_call_id;
    destination->payload.expected_fragment_count = copy->fragment_index;
    destination->payload.expected_token_cell_count = call->token_cell_count;
    destination->payload.stream_mask = call->stream_mask;
    destination->payload.final_writer_state = RUNTIME_STATE_WRITER_COMPLETE;
    destination->payload.layout_generation =
        counters->tensor_layout_event_count;
    destination->payload.backing_generation =
        counters->kv_backing_event_count;
    destination->payload.mutation_generation = counters->mutation_event_count;
    destination->payload.copy_byte_count = copy->copy_byte_count;
    destination->payload.reserve_failure_baseline =
        compute->payload_reserve_failure_baseline;
    destination->payload.reserve_failure_final =
        counters->payload_ringbuf_reserve_fail_count;
    destination->payload.writer_return_monotonic_ns =
        copy->writer_monotonic_ns;
    bpf_ringbuf_submit(destination, 0U);
    increment(&counters->state_unit_seal_submit_count);
    return true;
}

static __always_inline bool submit_state_unit_seal(
    const struct apply_call_state *call,
    const struct runtime_state_compute_call *compute,
    const struct state_unit_copy_loop_context *copy,
    struct runtime_state_bpf_counters *counters
) {
    struct runtime_state_kernel_state_unit_seal_event *destination;
    if (call == NULL || compute == NULL || copy == NULL || counters == NULL
        || copy->fragment_index == 0U || copy->copy_byte_count == 0U
        || compute->payload_reserve_failure_baseline
            != counters->payload_ringbuf_reserve_fail_count) {
        return false;
    }
    destination = bpf_ringbuf_reserve(
        &runtime_state_payload_events, sizeof(*destination), 0U);
    if (destination == NULL) {
        increment(&counters->payload_ringbuf_reserve_fail_count);
        return false;
    }
    destination->event_type = RUNTIME_STATE_KERNEL_STATE_UNIT_SEAL;
    destination->reserved = 0U;
    destination->payload.apply_call_id = call->apply_call_id;
    destination->payload.expected_fragment_count = copy->fragment_index;
    destination->payload.expected_token_cell_count =
        call->token_cell_count;
    destination->payload.stream_mask = call->stream_mask;
    destination->payload.final_writer_state =
        RUNTIME_STATE_WRITER_COMPLETE;
    destination->payload.layout_generation =
        counters->tensor_layout_event_count;
    destination->payload.backing_generation =
        counters->kv_backing_event_count;
    destination->payload.mutation_generation =
        counters->mutation_event_count;
    destination->payload.copy_byte_count = copy->copy_byte_count;
    destination->payload.reserve_failure_baseline =
        compute->payload_reserve_failure_baseline;
    destination->payload.reserve_failure_final =
        counters->payload_ringbuf_reserve_fail_count;
    destination->payload.writer_return_monotonic_ns =
        copy->writer_monotonic_ns;
    bpf_ringbuf_submit(destination, 0U);
    increment(&counters->state_unit_seal_submit_count);
    return true;
}

static __always_inline bool copy_writer_state_unit(
    const struct apply_call_state *call,
    const struct runtime_state_compute_call *compute,
    struct runtime_state_bpf_counters *counters
) {
    const __u32 zero = 0U;
    struct runtime_state_kernel_topology_event *topology;
    struct state_unit_copy_loop_context copy = {0};
    if (call == NULL || compute == NULL || counters == NULL) {
        return false;
    }
    topology = bpf_map_lookup_elem(&runtime_state_retained_topology, &zero);
    if (topology == NULL || topology->n_stream == 0U
        || topology->n_layer == 0U
        || topology->n_stream > RUNTIME_STATE_MAX_APPLY_STREAMS
        || topology->n_layer > RUNTIME_STATE_MAX_TENSOR_EVENTS
                / (2U * topology->n_stream)) {
        return false;
    }
    copy.apply_call_id = call->apply_call_id;
    copy.writer_monotonic_ns = bpf_ktime_get_ns();
    copy.expected_n_stream = topology->n_stream;
    copy.expected_n_layer = topology->n_layer;
    copy.v_trans = topology->v_trans;
    increment(&counters->active_emitter_count);
    if (((call->stream_mask & 1U) != 0U
            && (!run_flat_fragment_schedule(
                    &copy, call, 0U, RUNTIME_STATE_TENSOR_KIND_K)
                || !run_flat_fragment_schedule(
                    &copy, call, 0U, RUNTIME_STATE_TENSOR_KIND_V)))
        || ((call->stream_mask & 2U) != 0U
            && (!run_flat_fragment_schedule(
                    &copy, call, 1U, RUNTIME_STATE_TENSOR_KIND_K)
                || !run_flat_fragment_schedule(
                    &copy, call, 1U, RUNTIME_STATE_TENSOR_KIND_V)))
        || ((call->stream_mask & 4U) != 0U
            && (!run_flat_fragment_schedule(
                    &copy, call, 2U, RUNTIME_STATE_TENSOR_KIND_K)
                || !run_flat_fragment_schedule(
                    &copy, call, 2U, RUNTIME_STATE_TENSOR_KIND_V)))
        || ((call->stream_mask & 8U) != 0U
            && (!run_flat_fragment_schedule(
                    &copy, call, 3U, RUNTIME_STATE_TENSOR_KIND_K)
                || !run_flat_fragment_schedule(
                    &copy, call, 3U, RUNTIME_STATE_TENSOR_KIND_V)))
        || ((call->stream_mask & 16U) != 0U
            && (!run_flat_fragment_schedule(
                    &copy, call, 4U, RUNTIME_STATE_TENSOR_KIND_K)
                || !run_flat_fragment_schedule(
                    &copy, call, 4U, RUNTIME_STATE_TENSOR_KIND_V)))
        || ((call->stream_mask & 32U) != 0U
            && (!run_flat_fragment_schedule(
                    &copy, call, 5U, RUNTIME_STATE_TENSOR_KIND_K)
                || !run_flat_fragment_schedule(
                    &copy, call, 5U, RUNTIME_STATE_TENSOR_KIND_V)))
        || ((call->stream_mask & 64U) != 0U
            && (!run_flat_fragment_schedule(
                    &copy, call, 6U, RUNTIME_STATE_TENSOR_KIND_K)
                || !run_flat_fragment_schedule(
                    &copy, call, 6U, RUNTIME_STATE_TENSOR_KIND_V)))
        || ((call->stream_mask & 128U) != 0U
            && (!run_flat_fragment_schedule(
                    &copy, call, 7U, RUNTIME_STATE_TENSOR_KIND_K)
                || !run_flat_fragment_schedule(
                    &copy, call, 7U, RUNTIME_STATE_TENSOR_KIND_V)))
        || copy.failed != 0U) {
        (void)__sync_fetch_and_sub(&counters->active_emitter_count, 1U);
        return false;
    }
    {
        const bool sealed = submit_state_unit_seal(
            call, compute, &copy, counters);
        (void)__sync_fetch_and_sub(&counters->active_emitter_count, 1U);
        return sealed;
    }
}

static __always_inline bool copy_writer_state_unit_legacy(
    const struct apply_call_state *call,
    const struct runtime_state_compute_call *compute,
    struct runtime_state_bpf_counters *counters
) {
    return copy_writer_state_unit(call, compute, counters);
}

SEC("uretprobe")
int observe_process_ubatch_return(struct pt_regs *ctx) {
    const __u64 pid_tgid = bpf_get_current_pid_tgid();
    struct runtime_state_target_config *config = target_config(pid_tgid);
    struct runtime_state_bpf_counters *counters;
    struct runtime_state_bpf_diagnostics *diagnostics;
    struct runtime_state_compute_call *compute;
    struct apply_call_state *call;
    struct runtime_state_kernel_event *output;
    __u32 *ignored_depth;
    const __u64 result = PT_REGS_RC(ctx);
    struct state_unit_copy_loop_context planned_copy = {0};
    bool token_plan_complete;
    bool versions_complete;
    const bool gather_mode = gather_writer_boundary_mode(config);
    if (!writer_boundary_mode(config)) {
        return 0;
    }
    counters = global_counters();
    diagnostics = global_diagnostics();
    if (counters == NULL) {
        return 0;
    }
    ignored_depth = bpf_map_lookup_elem(
        &ignored_compute_returns_by_thread, &pid_tgid);
    if (ignored_depth != NULL) {
        if (*ignored_depth > 1U) {
            __u32 next_depth = *ignored_depth - 1U;
            (void)bpf_map_update_elem(
                &ignored_compute_returns_by_thread,
                &pid_tgid,
                &next_depth,
                BPF_ANY);
        } else {
            (void)bpf_map_delete_elem(
                &ignored_compute_returns_by_thread, &pid_tgid);
        }
        return 0;
    }
    compute = bpf_map_lookup_elem(&compute_calls_by_thread, &pid_tgid);
    if (compute == NULL) {
        increment(&counters->unmatched_call_count);
        if (diagnostics != NULL) {
            increment(&diagnostics->process_return_unmatched_count);
        }
        return 0;
    }
    call = bpf_map_lookup_elem(&apply_calls_by_thread, &pid_tgid);
    if (result == 0U) {
        if (diagnostics != NULL) {
            increment(&diagnostics->process_graph_compute_failed_count);
        }
        if (call != NULL) {
            restore_apply_streams(call);
            (void)delete_apply_token_plan(pid_tgid, call, false);
            (void)bpf_map_delete_elem(&apply_calls_by_thread, &pid_tgid);
        }
        (void)bpf_map_delete_elem(&compute_calls_by_thread, &pid_tgid);
        (void)__sync_fetch_and_sub(&counters->inflight_call_count, 1U);
        return 0;
    }
    if (call == NULL
        || compute->phase != RUNTIME_STATE_COMPUTE_APPLY_RETURNED
        || compute->apply_call_id == 0U
        || compute->apply_call_id != call->apply_call_id
        || compute->stream_mask != call->stream_mask
        || compute->token_cell_count != call->token_cell_count) {
        if (diagnostics != NULL) {
            increment(&diagnostics->process_apply_missing_count);
        }
        if (call != NULL) {
            restore_apply_streams(call);
            (void)delete_apply_token_plan(pid_tgid, call, false);
            (void)bpf_map_delete_elem(&apply_calls_by_thread, &pid_tgid);
        }
        (void)bpf_map_delete_elem(&compute_calls_by_thread, &pid_tgid);
        (void)__sync_fetch_and_sub(&counters->inflight_call_count, 1U);
        return 0;
    }
    if ((gather_mode
            && !plan_writer_state_unit(call, counters, &planned_copy))
        || (legacy_writer_boundary_mode(config)
            && !copy_writer_state_unit_legacy(call, compute, counters))) {
        restore_apply_streams(call);
        (void)delete_apply_token_plan(pid_tgid, call, false);
        (void)bpf_map_delete_elem(&apply_calls_by_thread, &pid_tgid);
        (void)bpf_map_delete_elem(&compute_calls_by_thread, &pid_tgid);
        (void)__sync_fetch_and_sub(&counters->inflight_call_count, 1U);
        return 0;
    }
    token_plan_complete = delete_apply_token_plan(pid_tgid, call, true);
    if (!token_plan_complete) {
        restore_apply_streams(call);
        increment(&counters->unmatched_call_count);
        if (diagnostics != NULL) {
            increment(&diagnostics->apply_state_conflict_count);
        }
        (void)bpf_map_delete_elem(&apply_calls_by_thread, &pid_tgid);
        (void)bpf_map_delete_elem(&compute_calls_by_thread, &pid_tgid);
        (void)__sync_fetch_and_sub(&counters->inflight_call_count, 1U);
        return 0;
    }
    versions_complete = complete_stream_version(call, 0U);
    versions_complete &= complete_stream_version(call, 1U);
    versions_complete &= complete_stream_version(call, 2U);
    versions_complete &= complete_stream_version(call, 3U);
    versions_complete &= complete_stream_version(call, 4U);
    versions_complete &= complete_stream_version(call, 5U);
    versions_complete &= complete_stream_version(call, 6U);
    versions_complete &= complete_stream_version(call, 7U);
    if (!versions_complete) {
        increment(&counters->unmatched_call_count);
        if (diagnostics != NULL) {
            increment(&diagnostics->apply_state_conflict_count);
        }
        (void)bpf_map_delete_elem(&apply_calls_by_thread, &pid_tgid);
        (void)bpf_map_delete_elem(&compute_calls_by_thread, &pid_tgid);
        (void)__sync_fetch_and_sub(&counters->inflight_call_count, 1U);
        return 0;
    }
    output = global_event_scratch();
    if (output == NULL) {
        restore_apply_streams(call);
        increment(&counters->unmatched_call_count);
        if (diagnostics != NULL) {
            increment(&diagnostics->apply_state_conflict_count);
        }
        (void)bpf_map_delete_elem(&apply_calls_by_thread, &pid_tgid);
        (void)bpf_map_delete_elem(&compute_calls_by_thread, &pid_tgid);
        (void)__sync_fetch_and_sub(&counters->inflight_call_count, 1U);
        return 0;
    }
    __builtin_memset(output, 0, sizeof(*output));
    output->event_type = RUNTIME_STATE_KERNEL_APPLY_COMPLETE;
    output->payload.apply_complete.apply_call_id = call->apply_call_id;
    output->payload.apply_complete.tgid = (__u32)(pid_tgid >> 32U);
    output->payload.apply_complete.tid = (__u32)pid_tgid;
    (void)submit_event(
        output, counters, &counters->apply_complete_event_count);
    if (gather_mode
        && !submit_state_unit_apply_ready(
            call, compute, &planned_copy, counters)) {
        increment(&counters->unmatched_call_count);
        if (diagnostics != NULL) {
            increment(&diagnostics->apply_state_conflict_count);
        }
        (void)bpf_map_delete_elem(&apply_calls_by_thread, &pid_tgid);
        (void)bpf_map_delete_elem(&compute_calls_by_thread, &pid_tgid);
        (void)__sync_fetch_and_sub(&counters->inflight_call_count, 1U);
        return 0;
    }
    if (gather_mode && bpf_send_signal_thread(SIGSTOP) != 0) {
        increment(&counters->unmatched_call_count);
        if (diagnostics != NULL) {
            increment(&diagnostics->apply_state_conflict_count);
        }
    }
    if (bpf_map_delete_elem(&apply_calls_by_thread, &pid_tgid) != 0
        || bpf_map_delete_elem(&compute_calls_by_thread, &pid_tgid) != 0) {
        increment(&counters->unmatched_call_count);
        if (diagnostics != NULL) {
            increment(&diagnostics->apply_state_delete_failure_count);
        }
    }
    (void)__sync_fetch_and_sub(&counters->inflight_call_count, 1U);
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
