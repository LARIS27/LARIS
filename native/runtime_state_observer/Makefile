CC ?= cc
CXX ?= c++
CLANG ?= clang-18
PKG_CONFIG ?= pkg-config
BUILD_DIR := build
CPPFLAGS := -Iinclude
CFLAGS := -std=c11 -Wall -Wextra -Werror -pedantic
CXXFLAGS := -std=c++17 -Wall -Wextra -Werror -pedantic
BPF_ARCH_INCLUDE ?= /usr/include/$(shell $(CC) -dumpmachine)
BPF_CFLAGS := -target bpf -D__TARGET_ARCH_x86 -D__BPF__ -O2 -g -Wall -Wextra -Werror \
	-Iinclude -I$(BPF_ARCH_INCLUDE)
LIBBPF_CFLAGS := $(shell $(PKG_CONFIG) --cflags libbpf 2>/dev/null)
LIBBPF_LIBS := $(shell $(PKG_CONFIG) --libs libbpf 2>/dev/null)

ifneq ($(strip $(LLAMA_SOURCE)),)
ABI_HEADERS := $(LLAMA_SOURCE)/src/llama-kv-cache.h \
	$(LLAMA_SOURCE)/src/llama-hparams.h $(LLAMA_SOURCE)/src/llama-batch.h \
	$(LLAMA_SOURCE)/ggml/include/ggml.h
ABI_HEADER_SHA256 := $(shell cat $(ABI_HEADERS) | sha256sum | cut -d' ' -f1)
ABI_FLAG_TEXT := $(CXXFLAGS)|src|include|ggml/include|ggml/src
ABI_FLAGS_SHA256 := $(shell printf '%s' '$(ABI_FLAG_TEXT)' | sha256sum | cut -d' ' -f1)
ABI_CPPFLAGS := $(CPPFLAGS) -DLARIS_ABI_QUALIFIED=1 \
	-DLARIS_ABI_HEADERS_SHA256='"$(ABI_HEADER_SHA256)"' \
	-DLARIS_ABI_FLAGS_SHA256='"$(ABI_FLAGS_SHA256)"' \
	-I$(LLAMA_SOURCE)/src -I$(LLAMA_SOURCE)/include \
	-I$(LLAMA_SOURCE)/ggml/include -I$(LLAMA_SOURCE)/ggml/src
endif

.PHONY: all fixture wire-check check-native-tools clean

all: $(BUILD_DIR)/runtime_state_observer $(BUILD_DIR)/runtime_state.bpf.o \
	$(BUILD_DIR)/wire_check $(BUILD_DIR)/abi_probe

fixture: $(BUILD_DIR)/libruntime_state_fixture.so \
	$(BUILD_DIR)/runtime_state_fixture_target \
	$(BUILD_DIR)/runtime_state_preexec_gate

wire-check: $(BUILD_DIR)/wire_check
	./$(BUILD_DIR)/wire_check

check-native-tools:
	@$(CC) -dumpfullversion | grep -Eq '^13(\.|$$)' || { echo "error: GCC 13.x is required" >&2; exit 1; }
	@$(CLANG) --version | head -n 1 | grep -Eq 'clang version 18\.' || { echo "error: Clang 18.x is required" >&2; exit 1; }
	@test "$$($(PKG_CONFIG) --modversion libbpf | cut -d. -f1,2)" = "1.3" || { echo "error: libbpf 1.3.x is required" >&2; exit 1; }

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/wire_check: tests/wire_check.c include/runtime_state_wire.h include/runtime_state_bpf_shared.h include/runtime_state_batch.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@

$(BUILD_DIR)/abi_probe: tools/abi_probe.cpp $(ABI_HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(ABI_CPPFLAGS) $< -o $@

$(BUILD_DIR)/runtime_state.bpf.o: src/runtime_state.bpf.c include/runtime_state_bpf_shared.h | $(BUILD_DIR) check-native-tools
	$(CLANG) $(BPF_CFLAGS) -c $< -o $@

$(BUILD_DIR)/runtime_state_observer: src/runtime_state_observer.c include/runtime_state_wire.h include/runtime_state_bpf_shared.h include/runtime_state_batch.h | $(BUILD_DIR) check-native-tools
	$(CC) $(CPPFLAGS) $(LIBBPF_CFLAGS) $(CFLAGS) $< -o $@ $(LIBBPF_LIBS)

$(BUILD_DIR)/libruntime_state_fixture.so: fixture/runtime_state_fixture.c include/runtime_state_fixture.h | $(BUILD_DIR) check-native-tools
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -shared $< -o $@

$(BUILD_DIR)/runtime_state_fixture_target: fixture/runtime_state_fixture_target.c include/runtime_state_fixture.h $(BUILD_DIR)/libruntime_state_fixture.so | $(BUILD_DIR) check-native-tools
	$(CC) $(CPPFLAGS) $(CFLAGS) -pthread $< -o $@ -L$(BUILD_DIR) \
		-Wl,-l:libruntime_state_fixture.so -Wl,-rpath,'$$ORIGIN'

$(BUILD_DIR)/runtime_state_preexec_gate: fixture/runtime_state_preexec_gate.c | $(BUILD_DIR) check-native-tools
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@

clean:
	$(RM) -r $(BUILD_DIR)
