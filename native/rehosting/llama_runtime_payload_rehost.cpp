#include "llama.h"

#include <algorithm>
#include <cerrno>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <inttypes.h>
#include <limits>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

struct Options {
    std::string model;
    std::string wrapped_state;
    std::string expected_libllama;
    std::string messages_file;
    int32_t destination_seq_id = -1;
    int32_t next_position = -1;
    uint32_t n_ctx = 8192;
    uint32_t n_parallel = 4;
    uint32_t n_batch = 128;
    uint32_t n_ubatch = 128;
    int32_t n_predict = 64;
    double temperature = 0.8;
    int32_t top_k = 40;
    double top_p = 0.95;
    double min_p = 0.05;
    uint32_t seed = 0;
    bool v_trans = false;
    bool v_trans_set = false;
};

void usage(const char * executable) {
    std::fprintf(
        stderr,
        "usage: %s --model MODEL --wrapped-state STATE --expected-libllama LIB "
        "--destination-seq-id ID "
        "--v-trans true|false "
        "--next-position POS --messages-file FILE "
        "[--ctx 8192 --parallel 4 --batch 128 --ubatch 128 --predict 64 "
        "--temperature 0.8 --top-k 40 --top-p 0.95 --min-p 0.05 --seed 0]\n",
        executable
    );
}

bool equals(const char * lhs, const char * rhs) {
    return std::strcmp(lhs, rhs) == 0;
}

bool parse_nonnegative(const char * text, int32_t & value) {
    try {
        std::size_t consumed = 0;
        const long long parsed = std::stoll(text, &consumed, 10);
        if (consumed != std::strlen(text) || parsed < 0 || parsed > INT32_MAX) {
            return false;
        }
        value = static_cast<int32_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_boolean(const char * text, bool & value) {
    if (equals(text, "true")) {
        value = true;
        return true;
    }
    if (equals(text, "false")) {
        value = false;
        return true;
    }
    return false;
}

bool parse_float(const char * text, double & value) {
    errno = 0;
    char * end = nullptr;
    const double parsed = std::strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !std::isfinite(parsed)) {
        return false;
    }
    value = parsed;
    return true;
}

bool parse_uint32(const char * text, uint32_t & value) {
    try {
        std::size_t consumed = 0;
        const unsigned long long parsed = std::stoull(text, &consumed, 10);
        if (consumed != std::strlen(text) || parsed > UINT32_MAX) return false;
        value = static_cast<uint32_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_options(int argc, char ** argv, Options & options) {
    for (int index = 1; index < argc; ++index) {
        const auto take_text = [&](std::string & destination) -> bool {
            if (index + 1 >= argc) {
                return false;
            }
            destination = argv[++index];
            return !destination.empty();
        };
        const auto take_uint = [&](uint32_t & destination) -> bool {
            int32_t parsed = 0;
            if (index + 1 >= argc || !parse_nonnegative(argv[++index], parsed) || parsed == 0) {
                return false;
            }
            destination = static_cast<uint32_t>(parsed);
            return true;
        };
        if (equals(argv[index], "--model")) {
            if (!take_text(options.model)) return false;
        } else if (equals(argv[index], "--wrapped-state")) {
            if (!take_text(options.wrapped_state)) return false;
        } else if (equals(argv[index], "--expected-libllama")) {
            if (!take_text(options.expected_libllama)) return false;
        } else if (equals(argv[index], "--messages-file")) {
            if (!take_text(options.messages_file)) return false;
        } else if (equals(argv[index], "--destination-seq-id")) {
            if (index + 1 >= argc || !parse_nonnegative(argv[++index], options.destination_seq_id)) return false;
        } else if (equals(argv[index], "--v-trans")) {
            if (index + 1 >= argc || !parse_boolean(argv[++index], options.v_trans)) return false;
            options.v_trans_set = true;
        } else if (equals(argv[index], "--next-position")) {
            if (index + 1 >= argc || !parse_nonnegative(argv[++index], options.next_position)) return false;
        } else if (equals(argv[index], "--ctx")) {
            if (!take_uint(options.n_ctx)) return false;
        } else if (equals(argv[index], "--parallel")) {
            if (!take_uint(options.n_parallel)) return false;
        } else if (equals(argv[index], "--batch")) {
            if (!take_uint(options.n_batch)) return false;
        } else if (equals(argv[index], "--ubatch")) {
            if (!take_uint(options.n_ubatch)) return false;
        } else if (equals(argv[index], "--predict")) {
            if (index + 1 >= argc || !parse_nonnegative(argv[++index], options.n_predict) || options.n_predict == 0) return false;
        } else if (equals(argv[index], "--temperature")) {
            if (index + 1 >= argc || !parse_float(argv[++index], options.temperature)) return false;
        } else if (equals(argv[index], "--top-k")) {
            if (index + 1 >= argc || !parse_nonnegative(argv[++index], options.top_k) || options.top_k == 0) return false;
        } else if (equals(argv[index], "--top-p")) {
            if (index + 1 >= argc || !parse_float(argv[++index], options.top_p)) return false;
        } else if (equals(argv[index], "--min-p")) {
            if (index + 1 >= argc || !parse_float(argv[++index], options.min_p)) return false;
        } else if (equals(argv[index], "--seed")) {
            if (index + 1 >= argc || !parse_uint32(argv[++index], options.seed)) return false;
        } else {
            return false;
        }
    }
    return !options.model.empty()
        && !options.wrapped_state.empty()
        && !options.expected_libllama.empty()
        && !options.messages_file.empty()
        && options.destination_seq_id >= 0
        && options.v_trans_set
        && static_cast<uint32_t>(options.destination_seq_id) < options.n_parallel
        && options.next_position > 0
        && options.n_ubatch <= options.n_batch
        && options.n_batch <= options.n_ctx
        && options.temperature > 0.0
        && options.top_p > 0.0 && options.top_p <= 1.0
        && options.min_p > 0.0 && options.min_p <= 1.0;
}

bool verify_mapped_libllama(const std::string & expected_path) {
    void * symbol = dlsym(RTLD_DEFAULT, "llama_state_seq_set_data_ext");
    Dl_info info{};
    struct stat expected_identity{};
    struct stat mapped_identity{};
    if (symbol == nullptr
            || dladdr(symbol, &info) == 0
            || info.dli_fname == nullptr
            || ::stat(expected_path.c_str(), &expected_identity) != 0
            || ::stat(info.dli_fname, &mapped_identity) != 0) {
        std::fprintf(stderr, "mapped libllama identity inspection failed\n");
        return false;
    }
    if (expected_identity.st_dev != mapped_identity.st_dev
            || expected_identity.st_ino != mapped_identity.st_ino) {
        std::fprintf(stderr, "mapped libllama differs from expected file\n");
        return false;
    }
    std::fprintf(
        stderr,
        "LARIS_META mapped_libllama_device=%ju\n",
        static_cast<uintmax_t>(mapped_identity.st_dev)
    );
    std::fprintf(
        stderr,
        "LARIS_META mapped_libllama_inode=%ju\n",
        static_cast<uintmax_t>(mapped_identity.st_ino)
    );
    return true;
}

std::string read_text(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    if (!input.good() && !input.eof()) {
        return {};
    }
    return buffer.str();
}

struct OwnedChatMessage {
    std::string role;
    std::string content;
};

void append_utf8(std::string & output, uint32_t point) {
    if (point <= 0x7FU) {
        output.push_back(static_cast<char>(point));
    } else if (point <= 0x7FFU) {
        output.push_back(static_cast<char>(0xC0U | (point >> 6U)));
        output.push_back(static_cast<char>(0x80U | (point & 0x3FU)));
    } else if (point <= 0xFFFFU) {
        output.push_back(static_cast<char>(0xE0U | (point >> 12U)));
        output.push_back(static_cast<char>(0x80U | ((point >> 6U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | (point & 0x3FU)));
    } else {
        output.push_back(static_cast<char>(0xF0U | (point >> 18U)));
        output.push_back(static_cast<char>(0x80U | ((point >> 12U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | ((point >> 6U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | (point & 0x3FU)));
    }
}

class JsonCursor {
public:
    explicit JsonCursor(const std::string & input) : input_(input) {}

    bool literal(const char * value) {
        const std::size_t length = std::strlen(value);
        if (input_.compare(offset_, length, value) != 0) return false;
        offset_ += length;
        return true;
    }

    bool string(std::string & output) {
        if (!literal("\"")) return false;
        output.clear();
        while (offset_ < input_.size()) {
            const unsigned char value = static_cast<unsigned char>(input_[offset_++]);
            if (value == '"') return true;
            if (value < 0x20U) return false;
            if (value != '\\') {
                output.push_back(static_cast<char>(value));
                continue;
            }
            if (offset_ >= input_.size()) return false;
            const char escape = input_[offset_++];
            switch (escape) {
                case '"': output.push_back('"'); break;
                case '\\': output.push_back('\\'); break;
                case '/': output.push_back('/'); break;
                case 'b': output.push_back('\b'); break;
                case 'f': output.push_back('\f'); break;
                case 'n': output.push_back('\n'); break;
                case 'r': output.push_back('\r'); break;
                case 't': output.push_back('\t'); break;
                case 'u': {
                    uint32_t point = 0;
                    if (!hex_quad(point)) return false;
                    if (point >= 0xD800U && point <= 0xDBFFU) {
                        if (!literal("\\u")) return false;
                        uint32_t low = 0;
                        if (!hex_quad(low) || low < 0xDC00U || low > 0xDFFFU) return false;
                        point = 0x10000U + ((point - 0xD800U) << 10U) + (low - 0xDC00U);
                    } else if (point >= 0xDC00U && point <= 0xDFFFU) {
                        return false;
                    }
                    append_utf8(output, point);
                    break;
                }
                default: return false;
            }
        }
        return false;
    }

    bool end() const {
        return offset_ == input_.size()
            || (offset_ + 1 == input_.size() && input_[offset_] == '\n');
    }

private:
    bool hex_quad(uint32_t & output) {
        if (offset_ + 4 > input_.size()) return false;
        output = 0;
        for (int index = 0; index < 4; ++index) {
            const char value = input_[offset_++];
            output <<= 4U;
            if (value >= '0' && value <= '9') output |= static_cast<uint32_t>(value - '0');
            else if (value >= 'a' && value <= 'f') output |= static_cast<uint32_t>(value - 'a' + 10);
            else return false;
        }
        return true;
    }

    const std::string & input_;
    std::size_t offset_ = 0;
};

bool parse_messages(
    const std::string & document,
    std::vector<OwnedChatMessage> & messages
) {
    JsonCursor cursor(document);
    if (!cursor.literal("{\"messages\":[")) return false;
    messages.clear();
    bool first = true;
    while (true) {
        if (cursor.literal("]")) break;
        if (!first && !cursor.literal(",")) return false;
        first = false;
        if (!cursor.literal("{\"content\":")) return false;
        OwnedChatMessage message;
        if (!cursor.string(message.content)
                || !cursor.literal(",\"role\":")
                || !cursor.string(message.role)
                || !cursor.literal("}")) {
            return false;
        }
        if ((message.role != "user" && message.role != "assistant")
                || message.content.empty()
                || message.content.find('\0') != std::string::npos
                || messages.size() >= 128) {
            return false;
        }
        messages.push_back(std::move(message));
    }
    return !messages.empty()
        && cursor.literal(",\"schema_version\":\"1.0\"}")
        && cursor.end();
}

template <typename Renderer>
bool render_chat_continuation(
    const std::vector<OwnedChatMessage> & owned,
    Renderer && renderer,
    std::string & continuation
) {
    if (owned.empty()) return false;
    const std::string assistant_boundary =
        "LARIS continuation boundary assistant";
    const std::vector<OwnedChatMessage> anchor{
        {"user", "LARIS continuation boundary user"},
        {"assistant", assistant_boundary},
    };
    std::vector<OwnedChatMessage> combined = anchor;
    combined.insert(combined.end(), owned.begin(), owned.end());

    std::string baseline;
    std::string complete;
    if (!renderer(anchor, false, baseline)
            || baseline.empty()
            || !renderer(combined, true, complete)) {
        return false;
    }
    const std::size_t boundary_offset = baseline.find(assistant_boundary);
    if (boundary_offset == std::string::npos
            || baseline.rfind(assistant_boundary) != boundary_offset) {
        return false;
    }
    const std::size_t cut = boundary_offset + assistant_boundary.size();
    if (complete.size() <= cut
            || complete.compare(0, cut, baseline, 0, cut) != 0) {
        return false;
    }
    continuation.assign(complete, cut, std::string::npos);
    return !continuation.empty();
}

bool render_chat(
    const llama_model * model,
    const std::vector<OwnedChatMessage> & owned,
    std::string & rendered
) {
    const char * chat_template = llama_model_chat_template(model, nullptr);
    if (chat_template == nullptr) return false;
    const auto renderer = [chat_template](
        const std::vector<OwnedChatMessage> & source,
        bool add_generation_prompt,
        std::string & output
    ) {
        std::vector<llama_chat_message> messages;
        messages.reserve(source.size());
        for (const auto & item : source) {
            messages.push_back({item.role.c_str(), item.content.c_str()});
        }
        const int32_t required = llama_chat_apply_template(
            chat_template,
            messages.data(),
            messages.size(),
            add_generation_prompt,
            nullptr,
            0
        );
        if (required <= 0) return false;
        std::vector<char> buffer(static_cast<std::size_t>(required) + 1U);
        const int32_t written = llama_chat_apply_template(
            chat_template,
            messages.data(),
            messages.size(),
            add_generation_prompt,
            buffer.data(),
            static_cast<int32_t>(buffer.size())
        );
        if (written <= 0 || written > required) return false;
        output.assign(buffer.data(), static_cast<std::size_t>(written));
        return !output.empty();
    };
    return render_chat_continuation(owned, renderer, rendered);
}

std::vector<uint8_t> read_binary(const std::string & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        return {};
    }
    const std::streamsize size = input.tellg();
    if (size <= 0 || static_cast<uint64_t>(size) > std::numeric_limits<std::size_t>::max()) {
        return {};
    }
    input.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(static_cast<std::size_t>(size));
    if (!input.read(reinterpret_cast<char *>(data.data()), size)) {
        return {};
    }
    return data;
}

std::vector<llama_token> tokenize(
    const llama_vocab * vocab,
    const std::string & text
) {
    const int32_t count = -llama_tokenize(
        vocab,
        text.c_str(),
        static_cast<int32_t>(text.size()),
        nullptr,
        0,
        false,
        true
    );
    if (count <= 0) {
        return {};
    }
    std::vector<llama_token> tokens(static_cast<std::size_t>(count));
    const int32_t written = llama_tokenize(
        vocab,
        text.c_str(),
        static_cast<int32_t>(text.size()),
        tokens.data(),
        count,
        false,
        true
    );
    if (written <= 0) {
        return {};
    }
    tokens.resize(static_cast<std::size_t>(written));
    return tokens;
}

bool decode(
    llama_context * context,
    const std::vector<llama_token> & tokens,
    llama_pos first_position,
    llama_seq_id sequence,
    bool logits_last
) {
    if (tokens.empty()) return false;
    llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
    batch.n_tokens = static_cast<int32_t>(tokens.size());
    for (int32_t index = 0; index < batch.n_tokens; ++index) {
        batch.token[index] = tokens[static_cast<std::size_t>(index)];
        batch.pos[index] = first_position + index;
        batch.n_seq_id[index] = 1;
        batch.seq_id[index][0] = sequence;
        batch.logits[index] = logits_last && index == batch.n_tokens - 1;
    }
    const int32_t result = llama_decode(context, batch);
    llama_batch_free(batch);
    if (result != 0) {
        std::fprintf(stderr, "decode failed: %d\n", result);
        return false;
    }
    return true;
}

bool decode_chunked(
    llama_context * context,
    const std::vector<llama_token> & tokens,
    llama_pos start_position,
    llama_seq_id sequence,
    uint32_t n_batch,
    int32_t & logical_batch_count
) {
    if (tokens.empty() || n_batch == 0) return false;
    logical_batch_count = 0;
    for (std::size_t offset = 0; offset < tokens.size(); offset += n_batch) {
        const std::size_t count = std::min<std::size_t>(n_batch, tokens.size() - offset);
        const bool logits_last = offset + count == tokens.size();
        const std::vector<llama_token> chunk(
            tokens.begin() + static_cast<std::ptrdiff_t>(offset),
            tokens.begin() + static_cast<std::ptrdiff_t>(offset + count)
        );
        if (!decode(
                context,
                chunk,
                start_position + static_cast<llama_pos>(offset),
                sequence,
                logits_last)) {
            return false;
        }
        ++logical_batch_count;
    }
    return true;
}

std::string piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buffer(256);
    int32_t written = llama_token_to_piece(
        vocab,
        token,
        buffer.data(),
        static_cast<int32_t>(buffer.size()),
        0,
        true
    );
    if (written < 0) {
        buffer.resize(static_cast<std::size_t>(-written));
        written = llama_token_to_piece(
            vocab,
            token,
            buffer.data(),
            static_cast<int32_t>(buffer.size()),
            0,
            true
        );
    }
    return written < 0
        ? std::string{}
        : std::string(buffer.data(), static_cast<std::size_t>(written));
}

bool generate(
    llama_context * context,
    const llama_vocab * vocab,
    llama_pos next_position,
    llama_seq_id sequence,
    int32_t n_predict,
    const Options & options
) {
    llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(options.top_k));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(static_cast<float>(options.top_p), 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_min_p(static_cast<float>(options.min_p), 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(static_cast<float>(options.temperature)));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(options.seed));
    bool success = true;
    int32_t generated_count = 0;
    for (int32_t index = 0; index < n_predict; ++index) {
        const llama_token token = llama_sampler_sample(sampler, context, -1);
        if (llama_vocab_is_eog(vocab, token)) break;
        llama_sampler_accept(sampler, token);
        const std::string text = piece(vocab, token);
        std::fwrite(text.data(), 1, text.size(), stdout);
        std::fflush(stdout);
        if (!decode(context, {token}, next_position++, sequence, true)) {
            success = false;
            break;
        }
        ++generated_count;
    }
    std::fputc('\n', stdout);
    std::fprintf(stderr, "LARIS_META generated_token_count=%d\n", generated_count);
    llama_sampler_free(sampler);
    return success;
}

}  // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    Options options;
    if (!parse_options(argc, argv, options)) {
        usage(argv[0]);
        return 2;
    }
    if (!verify_mapped_libllama(options.expected_libllama)) {
        return 2;
    }
    const std::string messages_document = read_text(options.messages_file);
    const std::vector<uint8_t> wrapped_state = read_binary(options.wrapped_state);
    std::vector<OwnedChatMessage> messages;
    if (!parse_messages(messages_document, messages) || wrapped_state.empty()) {
        std::fprintf(stderr, "messages or wrapped state is invalid or unreadable\n");
        return 2;
    }

    llama_log_set([](enum ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) std::fputs(text, stderr);
    }, nullptr);
    ggml_backend_load_all();
    llama_model_params model_parameters = llama_model_default_params();
    model_parameters.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(options.model.c_str(), model_parameters);
    if (model == nullptr) {
        std::fprintf(stderr, "model load failed\n");
        return 3;
    }
    std::string rendered;
    if (!render_chat(model, messages, rendered)) {
        std::fprintf(stderr, "GGUF_TEMPLATE_UNSUPPORTED\n");
        llama_model_free(model);
        return 6;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> rendered_tokens = tokenize(vocab, rendered);
    const llama_token bos = llama_vocab_bos(vocab);
    while (!rendered_tokens.empty() && rendered_tokens.front() == bos) {
        rendered_tokens.erase(rendered_tokens.begin());
    }
    const uint64_t per_sequence_capacity = options.n_ctx / options.n_parallel;
    const uint64_t required_capacity = static_cast<uint64_t>(options.next_position)
        + rendered_tokens.size()
        + static_cast<uint64_t>(options.n_predict);
    if (rendered_tokens.empty() || required_capacity > per_sequence_capacity) {
        std::fprintf(stderr, "CONTEXT_BUDGET_EXCEEDED\n");
        llama_model_free(model);
        return 5;
    }
    llama_context_params context_parameters = llama_context_default_params();
    context_parameters.n_ctx = options.n_ctx;
    context_parameters.n_batch = options.n_batch;
    context_parameters.n_ubatch = options.n_ubatch;
    context_parameters.n_seq_max = options.n_parallel;
    context_parameters.flash_attn_type = options.v_trans
        ? LLAMA_FLASH_ATTN_TYPE_DISABLED
        : LLAMA_FLASH_ATTN_TYPE_ENABLED;
    context_parameters.no_perf = true;
    llama_context * context = llama_init_from_model(model, context_parameters);
    if (context == nullptr) {
        std::fprintf(stderr, "context creation failed\n");
        llama_model_free(model);
        return 3;
    }

    const llama_seq_id sequence = options.destination_seq_id;
    const std::size_t consumed_bytes = llama_state_seq_set_data_ext(
        context,
        wrapped_state.data(),
        wrapped_state.size(),
        sequence,
        LLAMA_STATE_SEQ_FLAGS_NONE
    );
    const llama_pos restored_pos_min = llama_memory_seq_pos_min(
        llama_get_memory(context),
        sequence
    );
    const llama_pos restored_pos_max = llama_memory_seq_pos_max(
        llama_get_memory(context),
        sequence
    );
    if (consumed_bytes != wrapped_state.size()
            || restored_pos_min != 0
            || restored_pos_max != options.next_position - 1) {
        std::fprintf(stderr, "runtime-state restore validation failed\n");
        llama_free(context);
        llama_model_free(model);
        return 4;
    }
    std::fprintf(stderr, "LARIS_META restored_bytes=%zu\n", consumed_bytes);
    std::fprintf(stderr, "LARIS_META restored_pos_min=%d\n", restored_pos_min);
    std::fprintf(stderr, "LARIS_META restored_pos_max=%d\n", restored_pos_max);

    int32_t logical_batch_count = 0;
    if (!decode_chunked(
            context,
            rendered_tokens,
            options.next_position,
            sequence,
            options.n_batch,
            logical_batch_count)) {
        llama_free(context);
        llama_model_free(model);
        return 5;
    }
    const llama_pos generation_position = options.next_position
        + static_cast<llama_pos>(rendered_tokens.size());
    std::fprintf(stderr, "LARIS_META prompt_contract=gguf_default\n");
    std::fprintf(stderr, "LARIS_META appended_token_count=%zu\n", rendered_tokens.size());
    std::fprintf(stderr, "LARIS_META logical_batch_count=%d\n", logical_batch_count);
    std::fprintf(stderr, "LARIS_META effective_n_batch=%u\n", options.n_batch);
    std::fprintf(stderr, "LARIS_META effective_n_ubatch=%u\n", options.n_ubatch);
    std::fprintf(stderr, "LARIS_META sampler_temperature=%.9g\n", options.temperature);
    std::fprintf(stderr, "LARIS_META sampler_top_k=%d\n", options.top_k);
    std::fprintf(stderr, "LARIS_META sampler_top_p=%.9g\n", options.top_p);
    std::fprintf(stderr, "LARIS_META sampler_min_p=%.9g\n", options.min_p);
    std::fprintf(stderr, "LARIS_META sampler_seed=%u\n", options.seed);
    const bool generated = generate(
        context,
        vocab,
        generation_position,
        sequence,
        options.n_predict,
        options
    );
    llama_free(context);
    llama_model_free(model);
    return generated ? 0 : 5;
}
