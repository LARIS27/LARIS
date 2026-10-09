# Evaluation1 release metadata

This Git-tracked directory is the repository entry point for
`evaluation1-release-v1`. The external release contains four KV archives, one shared
data archive, and two synthetic fixture keys. It does not contain model weights.

## Access and integrity

[`artifact-locations.json`](artifact-locations.json) records the exact relative path,
byte size, SHA-256 digest, and file-specific public download URL of every external
file. The complete published directory is available from the
[Evaluation1 release folder](https://drive.google.com/drive/folders/1l_bvv0qQeiRossWd9HysLevyXwSn8HE2?usp=sharing).

[`release-manifest.json`](release-manifest.json) fixes the bundle inventory, model and
capture-mode association, archive and index digests, bundle counts, source revisions,
and model-license metadata used by the verifier.

## Licensing

The release contains derived KV-state artifacts and does not redistribute model
weights. Review the [third-party model notices](THIRD_PARTY_NOTICES.md) and the exact
upstream license texts included for
[Llama 3.2](licenses/LICENSE.llama-3.2.txt) and
[Qwen2.5](licenses/LICENSE.qwen2.5-apache-2.0.txt) before redistributing the release.

## Replay

Follow the [Evaluation1 release and replay guide](../../../../docs/reproducibility/evaluation1-release.md)
to stage and verify the external files, inspect the public bundle selectors, rehost a
selected record, and check recall against the documented Evaluation1 questions.

The key records are synthetic evaluation fixtures. They must never be reused
for a deployment or for non-synthetic evidence.
