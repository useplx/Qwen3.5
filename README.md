# Qwen3.5-2B for MyeLiN

This repository contains a PLeX port of Qwen3.5-2B for text, image, and video
input.

It does not claim authorship or ownership of Qwen3.5, its model architecture,
weights, name, or related materials.

## Supported features

- text, multiple-image, and video generation through the same entrypoint;
- native image decoding, resize, normalization, and patch preparation;
- the 24-layer vision encoder and visual merger;
- visual embedding fusion in the language decoder;
- grouped-query attention with QK normalization;
- ordered multimodal content and interleaved MRoPE for text, image, and video;
- chunked prompt processing;
- persistent decoder cache during a session;
- native byte-level BPE tokenization;
- configurable sampling;
- incremental UTF-8 output streaming.

Images are resized with the checkpoint's smart resize rules. Video input is a
sequence of already sampled encoded frames plus its sampling rate; odd frame
counts are padded as specified by the upstream processor. The visual graph uses
dynamic patch counts, so each media item keeps its own aspect ratio and token
count.

The package publishes the `generator` and `multimodal-generator` capabilities.

## Project status

This package does not produce a standalone `.mln`. A consuming application must
provide weights, assets, deployment policies, and a public entrypoint.

Generator integration is published through the manifest capability and does
not require a separate interface package.

## Requirements

- MyeLiN and SYNaPSe;
- Qwen3.5-2B-compatible weights;
- the matching `tokenizer.json`;
- PCRE2 and utf8proc for the native tokenizer;
- wasi-sdk for generation orchestration;
- selected quantization packages.

Weights and tokenizer assets are not distributed by this repository.

## Add the package

```toml
[dependencies]
"qwen3.5" = { git = "https://github.com/useplx/Qwen3_VL.git", version = "v1.0.0" }

[packages."qwen3.5"]
context = 1024
max_new_tokens = 224
prefill_chunk = 128
```

Add the official tokenizer to the root application's assets:

```text
assets/
└── tokenizer.json
```

## Package parts

| Name | Purpose |
|---|---|
| `qwen3.5:graph` | Model graph consumed by the SYNaPSe compiler. |
| `qwen3.5:orchestration` | Text and multimodal generation through public `forward`. |
| `qwen3.5:tokenizer` | Native BPE tokenizer provider. |
| `qwen3.5:vision` | Native image decoding and visual tensor preparation. |

A consuming application selects the package without rebuilding a wrapper:

```toml
[package]
name = "assistant"

[entrypoint]
package = "qwen"

[dependencies]
qwen = { git = "https://github.com/useplx/Qwen3_VL.git", version = "v1.0.0" }
```

## Public entrypoint

A standalone dependency exposes the package entrypoint declared by Qwen3.5:

```toml
[entrypoint]
package = "qwen3.5"
```

The public input may be UTF-8 prompt text or `Qwen3VL::Input`. Output is emitted
incrementally as UTF-8 and completes with the generated response.

```cpp
Qwen3VL::Input request{.content = {
    Qwen3VL::Text{"Compare these images: "},
    Qwen3VL::Image{first_encoded_image},
    Qwen3VL::Text{" and "},
    Qwen3VL::Image{second_encoded_image},
}};

session.invoke(
    "qwen3.5",
    request,
    [](std::string text) {
        std::print("{}", text);
    });
```

Argument serialization is selected automatically from the input type. There is
no public encode or decode call.

## Configuration

| Variable | Default | Meaning |
|---|---:|---|
| `context` | `1024` | Maximum prompt and generated sequence length. |
| `max_new_tokens` | `224` | Maximum tokens generated per invocation. |
| `prefill_chunk` | `128` | Prompt tokens processed per prefill step. |
| `do_sample` | `true` | Select sampling or greedy decoding. |
| `top_k` | `20` | Keep the highest scoring candidates; zero disables this filter. |
| `top_p` | `0.8` | Keep the smallest candidate prefix reaching this probability. |
| `temperature` | `0.7` | Scale logits before sampling. |
| `repetition_penalty` | `1.0` | Penalize token IDs already present in the sequence. |

The prompt and generated response must fit within `context`.

Sampling defaults follow the values currently defined by the port: top-k 20,
top-p 0.8, temperature 0.7, and repetition penalty 1.0.

## Useful policy paths

```text
model.language_model.embed_tokens.weight
model.language_model.layers.*.self_attn.q_proj.weight
model.language_model.layers.*.self_attn.k_proj.weight
model.language_model.layers.*.self_attn.v_proj.weight
model.language_model.layers.*.self_attn.o_proj.weight
model.language_model.layers.*.self_attn.q_norm.weight
model.language_model.layers.*.self_attn.k_norm.weight
model.language_model.layers.*.linear_attn.in_proj_qkv.weight
model.language_model.layers.*.linear_attn.in_proj_z.weight
model.language_model.layers.*.linear_attn.in_proj_b.weight
model.language_model.layers.*.linear_attn.in_proj_a.weight
model.language_model.layers.*.linear_attn.out_proj.weight
model.language_model.layers.*.linear_attn.A_log
model.language_model.layers.*.linear_attn.dt_bias
model.language_model.layers.*.mlp.gate_proj.weight
model.language_model.layers.*.mlp.up_proj.weight
model.language_model.layers.*.mlp.down_proj.weight
```

Use broad policies first, then override sensitive or incompatible parameters
with a more specific path.

## Quantization guidance

Start with the checkpoint's original BF16 storage and validate generation. Apply
lower-precision formats to large projection matrices only after establishing a
quality baseline. Keep small normalization weights at higher precision unless
the target has been validated otherwise.

The decoder cache currently uses F16 and should not be assigned an
immutable-weight codec.

See the [quantization guide](https://useplx.github.io/guide/quantize-weights/).

## Run

After the consuming project builds its artifact:

```bash
runner build/my-app.mln "Explain what a tensor is."
```

The example Sensix runner accepts repeated images and sampled video frames:

```bash
runner/build/sensix build/my-app.mln "Compare." \
    --image first.png --image second.png

runner/build/sensix build/my-app.mln "Describe the sequence." \
    --video-frame frame-0.png --video-frame frame-1.png --video-fps 2
```

A generation request produces multiple callback values as text becomes
available.

The package implements the checkpoint's inference architecture and common chat
generation path. It does not provide a media container decoder, batched
generation, training, or the Transformers tool-calling message API.

## Documentation

Full documentation lives at **[https://useplx.github.io/packages/qwen3.5/](https://useplx.github.io/packages/qwen3.5/)**.

- [Package guide](https://useplx.github.io/packages/qwen3.5/)
- [Quantization guide](https://useplx.github.io/guide/quantize-weights/)

## License

This source port follows the upstream
[Apache License 2.0](LICENSE). Users must separately review the terms attached to
the weights, datasets, tokenizer files, names, and trademarks they use.
