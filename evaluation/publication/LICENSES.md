# Evaluation model provenance and notices

This file records only the model notices needed to identify the evaluated artifacts and
to construct a reviewed external release. LARIS source code has its own repository
license. Model weights are not redistributed by this repository, release, or Git branch.
Users obtain the models from the pinned upstream sources and verify the recorded
artifact digests in `evaluation/experiment_setup/models.yaml`.

KV state is treated conservatively as a model-derived artifact. The packager therefore
requires a separate human `PUBLIC_APPROVED` decision for each model before it will create
a distributable KV release. This document records provenance and notice obligations; it
does not itself grant that approval or replace legal review.

## Llama 3.2 3B Instruct

- LARIS model ID: `llama-3.2-3b-instruct-q4-k-m`
- Upstream: `meta-llama/Llama-3.2-3B-Instruct`
- Pinned revision: `0cb88a4f764b7a12671c53f0838cd831a0843b95`
- License ID: `llama-3.2-community`
- License: [Llama 3.2 Community License](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/blob/main/LICENSE.txt)
- Acceptable use policy: <https://www.llama.com/llama3/use-policy/>
- Required notice: [`NOTICE.llama-3.2.txt`](NOTICE.llama-3.2.txt)

## Qwen2.5 7B Instruct GGUF

- LARIS model ID: `qwen2.5-7b-instruct-q4-k-m`
- Upstream: `Qwen/Qwen2.5-7B-Instruct-GGUF`
- Pinned revision: `bb5d59e06d9551d752d08b292a50eb208b07ab1f`
- License ID: `Apache-2.0`
- License: [Apache License 2.0](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/blob/main/LICENSE)

The external release includes the exact upstream license files only after the explicit
model-level review gate succeeds. No extra attribution is invented beyond the upstream
license and the model identity needed for reproducibility.
