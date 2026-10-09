CXX ?= c++
LLAMA_SOURCE ?=
LLAMA_BUILD ?=
BUILD_DIR ?= build

TARGET := $(BUILD_DIR)/llama-runtime-payload-rehost
SOURCE := llama_runtime_payload_rehost.cpp
CHECK_TARGET := $(BUILD_DIR)/rehosting-chat-continuation-check
CHECK_SOURCE := ../../tests/native/rehosting_chat_continuation_check.cpp

.PHONY: all check clean

all: $(TARGET)

$(TARGET): $(SOURCE)
	@test -n "$(LLAMA_SOURCE)" || (echo "LLAMA_SOURCE is required" >&2; exit 2)
	@test -n "$(LLAMA_BUILD)" || (echo "LLAMA_BUILD is required" >&2; exit 2)
	mkdir -p $(BUILD_DIR)
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror \
		-I$(LLAMA_SOURCE)/include -I$(LLAMA_SOURCE)/ggml/include \
		$(SOURCE) -L$(LLAMA_BUILD)/bin -Wl,-rpath-link,$(LLAMA_BUILD)/bin \
		-lllama -lggml -ldl -o $(TARGET)

check: $(CHECK_TARGET)
	$(CHECK_TARGET)

$(CHECK_TARGET): $(SOURCE) $(CHECK_SOURCE)
	@test -n "$(LLAMA_SOURCE)" || (echo "LLAMA_SOURCE is required" >&2; exit 2)
	@test -n "$(LLAMA_BUILD)" || (echo "LLAMA_BUILD is required" >&2; exit 2)
	mkdir -p $(BUILD_DIR)
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror \
		-I$(LLAMA_SOURCE)/include -I$(LLAMA_SOURCE)/ggml/include \
		$(CHECK_SOURCE) -L$(LLAMA_BUILD)/bin -Wl,-rpath-link,$(LLAMA_BUILD)/bin \
		-lllama -lggml -ldl -o $(CHECK_TARGET)

clean:
	rm -f $(TARGET) $(CHECK_TARGET)
