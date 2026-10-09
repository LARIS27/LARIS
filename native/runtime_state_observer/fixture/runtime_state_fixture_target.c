#define _POSIX_C_SOURCE 200809L

#include "runtime_state_fixture.h"

#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static int cache_marker;
static int hparams_marker;
static pthread_t blocked_apply_thread;
static bool blocked_apply_running;

static void run_observation(
    bool use_preloaded_cache,
    bool invalid_batch_shape,
    bool token_count_exceeds
) {
    uint32_t streams[] = {1U, 3U};
    uint32_t first_cells[] = {11U, 3U, 19U};
    uint32_t second_cells[] = {12U, 4U, 20U};
    struct runtime_state_fixture_vector cell_vectors[] = {
        {first_cells, first_cells + 3U, first_cells + 3U},
        {second_cells, second_cells + 3U, second_cells + 3U},
    };
    struct runtime_state_fixture_slot_info slot = {
        .s0 = 0U,
        .s1 = 1U,
        .strm = {streams, streams + 2U, streams + 2U},
        .idxs = {cell_vectors, cell_vectors + 2U, cell_vectors + 2U},
    };
    int32_t positions[] = {4, 7, 8, 2, 5, 9};
    int32_t sequence_counts[] = {1, 1, 1, 1, 1, 1};
    int32_t sequence_values[] = {2, 2, 2, 0, 0, 0};
    int32_t *sequence_pointers[] = {
        &sequence_values[0], &sequence_values[1], &sequence_values[2],
        &sequence_values[3], &sequence_values[4], &sequence_values[5],
    };
    struct runtime_state_fixture_ubatch ubatch = {
        .b_equal_seqs = 1U,
        .n_tokens = 6U,
        .n_seq_tokens = 3U,
        .n_seqs = 2U,
        .n_seqs_unq = 2U,
        .n_pos = 1U,
        .pos = positions,
        .n_seq_id = sequence_counts,
        .seq_id = sequence_pointers,
    };
    if (invalid_batch_shape) {
        ubatch.n_tokens = 5U;
    }
    if (token_count_exceeds) {
        ubatch.n_tokens = 2049U;
    }
    void *cache_object = &cache_marker;
    if (use_preloaded_cache) {
        cache_object = runtime_state_fixture_preloaded_cache();
    } else {
        runtime_state_fixture_kv_constructor(
            &cache_marker,
            NULL,
            &hparams_marker,
            1,
            1,
            true,
            false,
            false,
            RUNTIME_STATE_FIXTURE_KV_SIZE,
            RUNTIME_STATE_FIXTURE_N_STREAM);
        if (runtime_state_fixture_n_pos_per_embd(&hparams_marker) != 1U) {
            return;
        }
    }
    runtime_state_fixture_apply_ubatch(cache_object, &slot, &ubatch);
}

static void run_large_state_restore(void) {
    uint32_t stream = 1U;
    uint32_t cells[176];
    int32_t positions[176];
    int32_t sequence_counts[176];
    int32_t sequence_value = 2;
    int32_t *sequence_pointers[176];
    struct runtime_state_fixture_vector cell_vector = {
        cells, cells + 176U, cells + 176U,
    };
    struct runtime_state_fixture_slot_info slot = {
        .s0 = 0U,
        .s1 = 0U,
        .strm = {&stream, &stream + 1U, &stream + 1U},
        .idxs = {&cell_vector, &cell_vector + 1U, &cell_vector + 1U},
    };
    struct runtime_state_fixture_ubatch ubatch = {
        .b_equal_seqs = 1U,
        .n_tokens = 176U,
        .n_seq_tokens = 176U,
        .n_seqs = 1U,
        .n_seqs_unq = 1U,
        .n_pos = 1U,
        .pos = positions,
        .n_seq_id = sequence_counts,
        .seq_id = sequence_pointers,
    };
    uint32_t index;

    for (index = 0U; index < 176U; ++index) {
        cells[index] = index;
        positions[index] = (int32_t)index;
        sequence_counts[index] = 1;
        sequence_pointers[index] = &sequence_value;
    }
    runtime_state_fixture_apply_ubatch(
        runtime_state_fixture_preloaded_cache(), &slot, &ubatch);
}

static void run_variable_stream_observation(void) {
    uint32_t streams[] = {1U, 3U};
    uint32_t first_cells[] = {11U, 3U};
    uint32_t second_cells[] = {12U, 4U, 20U};
    struct runtime_state_fixture_vector cell_vectors[] = {
        {first_cells, first_cells + 2U, first_cells + 2U},
        {second_cells, second_cells + 3U, second_cells + 3U},
    };
    int32_t positions[] = {0, 1, 2, 3, 4};
    int32_t sequence_counts[] = {1, 1, 1, 1, 1};
    int32_t sequence_values[] = {2, 2, 0, 0, 0};
    int32_t *sequence_pointers[] = {
        &sequence_values[0],
        &sequence_values[1],
        &sequence_values[2],
        &sequence_values[3],
        &sequence_values[4],
    };
    struct runtime_state_fixture_slot_info slot = {
        .s0 = 0U,
        .s1 = 0U,
        .strm = {streams, streams + 2U, streams + 2U},
        .idxs = {cell_vectors, cell_vectors + 2U, cell_vectors + 2U},
    };
    struct runtime_state_fixture_ubatch ubatch = {
        .b_equal_seqs = 0U,
        .n_tokens = 5U,
        .n_seq_tokens = 0U,
        .n_seqs = 2U,
        .n_seqs_unq = 2U,
        .n_pos = 1U,
        .pos = positions,
        .n_seq_id = sequence_counts,
        .seq_id = sequence_pointers,
    };

    runtime_state_fixture_apply_ubatch(
        runtime_state_fixture_preloaded_cache(), &slot, &ubatch);
}

static void *run_blocked_apply(void *unused) {
    (void)unused;
    run_observation(true, false, false);
    return NULL;
}

static bool start_blocked_apply(void) {
    struct timespec delay = {.tv_sec = 0, .tv_nsec = 1000000L};
    unsigned int attempt;
    if (blocked_apply_running) {
        return false;
    }
    runtime_state_fixture_pause_next_apply();
    if (pthread_create(&blocked_apply_thread, NULL, run_blocked_apply, NULL) != 0) {
        runtime_state_fixture_release_apply();
        return false;
    }
    blocked_apply_running = true;
    for (attempt = 0U; attempt < 5000U; ++attempt) {
        if (runtime_state_fixture_apply_is_paused()) {
            return true;
        }
        (void)nanosleep(&delay, NULL);
    }
    runtime_state_fixture_release_apply();
    (void)pthread_join(blocked_apply_thread, NULL);
    blocked_apply_running = false;
    return false;
}

static bool release_blocked_apply(void) {
    if (!blocked_apply_running) {
        return false;
    }
    runtime_state_fixture_release_apply();
    if (pthread_join(blocked_apply_thread, NULL) != 0) {
        return false;
    }
    blocked_apply_running = false;
    return true;
}

static bool run_process_lifecycle(
    uint32_t write_generation,
    int graph_status,
    const char *oracle_label
) {
    uint32_t stream = 1U;
    uint32_t cell = 11U;
    struct runtime_state_fixture_vector cell_vector = {
        &cell, &cell + 1U, &cell + 1U,
    };
    struct runtime_state_fixture_slot_info slot = {
        .s0 = 0U,
        .s1 = 0U,
        .strm = {&stream, &stream + 1U, &stream + 1U},
        .idxs = {&cell_vector, &cell_vector + 1U, &cell_vector + 1U},
    };
    int32_t position = 11;
    int32_t sequence_count = 1;
    int32_t sequence_value = 2;
    int32_t *sequence_pointer = &sequence_value;
    struct runtime_state_fixture_ubatch ubatch = {
        .b_equal_seqs = 1U,
        .n_tokens = 1U,
        .n_seq_tokens = 1U,
        .n_seqs = 1U,
        .n_seqs_unq = 1U,
        .n_pos = 1U,
        .pos = &position,
        .n_seq_id = &sequence_count,
        .seq_id = &sequence_pointer,
    };
    struct runtime_state_fixture_context context = {
        .cache_object = runtime_state_fixture_preloaded_cache(),
        .slot = &slot,
    };
    struct runtime_state_fixture_memory_context memory = {
        .graph_status = graph_status,
        .write_generation = write_generation,
    };
    int status = -1;
    void *result = runtime_state_fixture_process_ubatch(
        &context, &ubatch, &memory, &status);

    if (graph_status != 0) {
        return result == NULL && status == graph_status;
    }
    if (result != &context || status != 0) {
        return false;
    }
    (void)printf(
        "ORACLE process=%s key=%u value=%u\n",
        oracle_label,
        (unsigned int)runtime_state_fixture_oracle_byte(
            stream, 0U, cell, false),
        (unsigned int)runtime_state_fixture_oracle_byte(
            stream, 0U, cell, true));
    fflush(stdout);
    return true;
}

static void run_prepare_lifecycle(void) {
    uint32_t stream = 1U;
    uint32_t cell = 11U;
    struct runtime_state_fixture_vector cell_vector = {
        &cell, &cell + 1U, &cell + 1U,
    };
    struct runtime_state_fixture_slot_info slot = {
        .strm = {&stream, &stream + 1U, &stream + 1U},
        .idxs = {&cell_vector, &cell_vector + 1U, &cell_vector + 1U},
    };
    struct runtime_state_fixture_ubatch ubatch = {
        .b_equal_seqs = 1U,
        .n_tokens = 1U,
        .n_seq_tokens = 1U,
        .n_seqs = 1U,
        .n_seqs_unq = 1U,
    };
    struct runtime_state_fixture_context context = {
        .cache_object = runtime_state_fixture_preloaded_cache(),
        .slot = &slot,
    };
    runtime_state_fixture_prepare_ubatch(&context, &ubatch);
}

int main(void) {
    char line[128];
    (void)runtime_state_fixture_preloaded_cache();
    printf("READY %ld\n", (long)getpid());
    fflush(stdout);
    while (fgets(line, sizeof(line), stdin) != NULL) {
        if (strcmp(line, "run\n") == 0) {
            run_observation(false, false, false);
            puts("DONE");
            fflush(stdout);
        } else if (strcmp(line, "apply-preloaded\n") == 0) {
            run_observation(true, false, false);
            puts("DONE");
            fflush(stdout);
        } else if (strcmp(line, "apply-invalid-batch\n") == 0) {
            run_observation(true, true, false);
            puts("DONE");
            fflush(stdout);
        } else if (strcmp(line, "apply-token-count-exceeds\n") == 0) {
            run_observation(true, false, true);
            puts("DONE");
            fflush(stdout);
        } else if (strcmp(line, "apply-large-state-restore\n") == 0) {
            run_large_state_restore();
            puts("DONE");
            fflush(stdout);
        } else if (strcmp(line, "apply-variable-streams\n") == 0) {
            run_variable_stream_observation();
            puts("DONE");
            fflush(stdout);
        } else if (strcmp(line, "prepare-only\n") == 0) {
            run_prepare_lifecycle();
            puts("DONE");
            fflush(stdout);
        } else if (strcmp(line, "process-once\n") == 0) {
            if (!run_process_lifecycle(1U, 0, "1")) {
                fputs("ERROR process lifecycle failed\n", stderr);
                return 2;
            }
            puts("DONE");
            fflush(stdout);
        } else if (strcmp(line, "process-overwrite\n") == 0) {
            struct runtime_state_fixture_memory_context memory = {
                .graph_status = 0,
                .write_generation = 1U,
            };
            if (!run_process_lifecycle(memory.write_generation, 0, "1")) {
                fputs("ERROR first process lifecycle failed\n", stderr);
                return 2;
            }
            memory.write_generation = 2U;
            if (!run_process_lifecycle(memory.write_generation, 0, "2")) {
                fputs("ERROR second process lifecycle failed\n", stderr);
                return 2;
            }
            puts("DONE");
            fflush(stdout);
        } else if (strcmp(line, "process-failed\n") == 0) {
            if (!run_process_lifecycle(3U, 7, "failed")) {
                fputs("ERROR failed process lifecycle completed\n", stderr);
                return 2;
            }
            puts("DONE");
            fflush(stdout);
        } else if (strcmp(line, "start-blocked-apply\n") == 0) {
            if (!start_blocked_apply()) {
                fputs("ERROR blocked apply did not start\n", stderr);
                return 2;
            }
            puts("APPLY_BLOCKED");
            fflush(stdout);
        } else if (strcmp(line, "release-blocked-apply\n") == 0) {
            if (!release_blocked_apply()) {
                fputs("ERROR blocked apply did not finish\n", stderr);
                return 2;
            }
            puts("DONE");
            fflush(stdout);
        } else if (strcmp(line, "sequence-remove\n") == 0) {
            runtime_state_fixture_sequence_remove();
            puts("DONE");
            fflush(stdout);
        } else if (strcmp(line, "exit\n") == 0) {
            if (blocked_apply_running) {
                (void)release_blocked_apply();
            }
            return 0;
        } else {
            fputs("ERROR unknown command\n", stderr);
            return 2;
        }
    }
    return 0;
}
