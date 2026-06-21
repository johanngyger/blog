# A local coding agent on a fanless laptop

*21 June 2026*

Ever since coding agents became genuinely useful, the honest answer to "can I run one
entirely on my laptop?" has been *not really*. Local models were either too weak to be
trusted with real edits, or too slow to be pleasant, and usually both. That has quietly
changed. Between small mixture-of-experts models, quantization-aware training, and
speculative decoding landing in `llama.cpp`, a 32 GiB laptop can now run an agent that
is genuinely good enough to use.

This is a writeup of getting one working on a **MacBook Air M5 with 32 GiB of unified
memory**, end to end and copy-pasteable, plus the benchmarks that show which pieces
actually pull their weight.

## The short answer

Yes, with caveats. On this hardware you get roughly **60 tokens/second** of generation
and **~700 tokens/second** of prompt processing from a 26B model, which is comfortably
interactive for an agent loop. It is not a frontier model and it is not workstation
speed, but it reads code, edits files, runs commands, and reasons about what it is
doing, locally, offline, for free.

## The stack

| Piece | Choice |
|---|---|
| Inference server | [`llama.cpp`](https://github.com/ggml-org/llama.cpp) (`llama-server`) |
| Model | Gemma 4 26B-A4B, Google's **QAT** weights packaged as an Unsloth Q4_K_XL GGUF (~14 GB) |
| Speedup | MTP draft model for speculative decoding (~250 MB) |
| Coding agent | [`pi`](https://pi.dev) talking to the server over its OpenAI-compatible API |

Gemma 4 26B-A4B is a mixture-of-experts model: 26B total parameters but only ~4B active
per token. That is the whole reason this works on a memory-bandwidth-limited laptop,
each token only has to read ~4B parameters from memory, not 26B.

## The machine, and why it can run this at all

The laptop is a **MacBook Air M5**: a 10-core CPU, a **10-core GPU**, **32 GiB of unified
memory**, and roughly **150 GB/s of memory bandwidth**. No discrete GPU, no fan. A few
things about Apple Silicon make local inference work here that would not on a comparable
PC laptop:

- **Unified memory.** The CPU and GPU share one physical pool of RAM. There is no separate
  VRAM and no copying weights across a PCIe bus, the GPU addresses all 32 GiB directly. So
  the laptop's RAM *is* the GPU's memory: a 14 GB model plus its KV cache simply lives in
  that pool and the GPU reads it in place. On a PC you would need a discrete GPU with
  enough VRAM to hold the whole model; here it comes for free.
- **Metal does the heavy lifting.** `-ngl 999` offloads every layer to the GPU through
  Apple's Metal backend, so the matmuls run on those 10 GPU cores, not the CPU.
- **M5's fast prompt processing.** Feeding the model your system prompt and file context
  each turn is compute-bound, and this setup processes it at roughly **700 tok/s**. A year
  or two ago, prefill over a big context was the painful part on a laptop; here it is fast
  enough to stay out of the way.

The one thing the hardware cannot cheat is **memory bandwidth**. Generating each token
means reading the active weights out of memory, and at ~150 GB/s that sets a hard ceiling,
about 60 tok/s for this model. Prompt processing is compute-bound, while generation is
bandwidth-bound, so additional compute does not raise that generation ceiling. That single
fact explains every model decision below: pick a model that reads *few bytes per token*
(a mixture-of-experts with a small active set), and use speculative decoding to verify
several tokens per memory sweep.

So "is it feasible?" is really four questions, and this hardware answers each differently:

- **Does it fit?** Bounded by 32 GiB of unified memory.
- **Is it fast enough?** Generation is bounded by ~150 GB/s bandwidth; prefill reaches
  roughly 700 tok/s.
- **Is it good enough?** Bounded by whatever quality survives at a quant that fits.
- **Is the resource cost acceptable?** It has to leave headroom for macOS, or it swaps and
  the whole thing falls apart.

The Gemma setup below clears all four. The "Choosing the model" section later is what
happens when a model does not.

## Setup

Everything lives under one directory, here `~/local-ai` (put it wherever you like): a
shared `llama.cpp` build, the model weights, and a small Python env for downloads.

```
~/local-ai/
├── .venv/                # huggingface_hub + hf_xet (model downloads)
├── engines/
│   └── llama.cpp/        # the build; binary at engines/llama.cpp/build/bin/llama-server
└── models/
    └── unsloth-gemma-4-26B-A4B-it-qat-GGUF/
```

### 1. Build llama.cpp

```sh
mkdir -p ~/local-ai/engines
git clone https://github.com/ggml-org/llama.cpp ~/local-ai/engines/llama.cpp
cmake -S ~/local-ai/engines/llama.cpp -B ~/local-ai/engines/llama.cpp/build \
  -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_ACCELERATE=ON \
  -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple
cmake --build ~/local-ai/engines/llama.cpp/build --config Release -j
```

This guide leans on [How to setup a local coding agent on
macOS](https://ikyle.me/blog/2026/how-to-setup-a-local-coding-agent-on-macos) and
[Unsloth's llama.cpp MTP guide](https://unsloth.ai/docs/models/mtp#llama.cpp-mtp-guide)
for the toolchain details.

### 2. A Python env for downloads

```sh
python3.11 -m venv ~/local-ai/.venv
~/local-ai/.venv/bin/pip install -U huggingface_hub hf_xet
```

### 3. Download the model

The main QAT quant plus its matching MTP draft (the draft lives in an `MTP/` subfolder).
The `mmproj-*` vision files in the same repo are not needed for a text-only coding agent.

One honest note on the source. The QAT itself is **Google's**, and Google ships its own
official Q4_0 GGUF ([`google/gemma-4-26B-A4B-it-qat-q4_0-gguf`](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf)).
This setup uses Unsloth's dynamic Q4_K_XL because the same repository includes the
**MTP draft**, not because this post establishes it as a quality upgrade. Google's GGUF
repository ships no ready llama.cpp draft, and that draft is what buys the ~1.5x
speculative speedup below.

```sh
~/local-ai/.venv/bin/hf download unsloth/gemma-4-26B-A4B-it-qat-GGUF \
  gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf \
  MTP/mtp-gemma-4-26B-A4B-it.gguf \
  --local-dir ~/local-ai/models/unsloth-gemma-4-26B-A4B-it-qat-GGUF
```

### 4. The launch script

Save the server invocation as a command in any directory on your `PATH` (here
`~/.local/bin`):

```sh
mkdir -p ~/.local/bin
cat > ~/.local/bin/gemma-serve <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

LLAMA=~/local-ai/engines/llama.cpp/build/bin/llama-server
MODELS=~/local-ai/models/unsloth-gemma-4-26B-A4B-it-qat-GGUF

exec "$LLAMA" \
  -m "$MODELS/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf" \
  --model-draft "$MODELS/MTP/mtp-gemma-4-26B-A4B-it.gguf" \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  -ngl 999 \
  -fa on \
  -ctk q8_0 -ctv q8_0 \
  -c 65536 \
  --parallel 1 \
  --reasoning on \
  --reasoning-format deepseek \
  --temp 1.0 --top-p 0.95 --top-k 64 --samplers "temperature;top_p;top_k" \
  --host 127.0.0.1 --port 8080
EOF
chmod +x ~/.local/bin/gemma-serve
```

The non-obvious flags:

- **`--model-draft` / `--spec-type draft-mtp` / `--spec-draft-n-max`** enable MTP
  speculative decoding. A tiny draft head proposes the next few tokens and the main
  model verifies them in parallel. The method is designed to preserve model quality; in
  the greedy benchmark below it produced identical output (more on the size of that win
  below).
- **`-ctk q8_0 -ctv q8_0`** quantize the KV cache to 8-bit, which roughly halves its
  memory and is near-lossless. It only pays off with flash attention (**`-fa on`**),
  otherwise the cache gets dequantized on every attention step.
- **`--reasoning-format deepseek`** routes the model's chain-of-thought into a separate
  `reasoning_content` field instead of leaking `<think>...</think>` tags into the reply.
  Agents that understand reasoning models expect this.
- **`--temp 1.0 --top-p 0.95 --top-k 64`** are Gemma 4's
  [recommended samplers](https://unsloth.ai/docs/models/gemma-4). Counter-intuitively,
  [Gemma 4 wants a high temperature even for coding](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/discussions/21),
  the opposite of the usual "turn temperature down for code" advice. The sampler order
  matters too.

### 5. Point the agent at it

Install [`pi`](https://pi.dev), then define a custom OpenAI-compatible provider in
`~/.pi/agent/models.json`:

```jsonc
{
  "providers": {
    "gemma4-local": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "authHeader": false,
      "compat": {
        "supportsDeveloperRole": false,    // server has no "developer" role
        "supportsReasoningEffort": false   // server ignores reasoning_effort
      },
      "models": [
        {
          "id": "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
          "reasoning": true,               // read the reasoning_content channel
          "input": ["text"],               // text-only (drop the mmproj for images)
          "contextWindow": 65536,
          "maxTokens": 16384
        }
      ]
    }
  }
}
```

Then run it:

```sh
gemma-serve                          # start the server (127.0.0.1:8080)
pi --provider gemma4-local           # in another shell, point the agent at it
```

Two pairings worth understanding:

- The `compat` block exists because a bare `llama.cpp` server is not a perfect OpenAI
  clone. Telling the client not to send a `developer` role or a `reasoning_effort` field
  avoids errors.
- `"reasoning": true` makes the agent read the separate thinking channel that
  `--reasoning-format deepseek` produces on the server side. They are a pair. And the
  agent sends no sampling params of its own, so the server-side samplers above are what
  actually take effect, set them there, not in the client.

See [pi's custom-provider docs](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/models.md)
for the full field list.

## Is speculative decoding actually worth it?

Yes, and the size of the win surprised me. Same prompt, greedy decoding, identical
output every run, so tokens/second is directly comparable:

| Config | Generation | vs. no draft | Draft acceptance |
|---|---|---|---|
| no draft model | 39.7 tok/s | 1.00x | – |
| MTP, n-max 1 | 59.5 tok/s | 1.50x | 92% |
| MTP, n-max 2 | 60.3 tok/s | 1.52x | 87% |
| MTP, n-max 3 | 58.3 tok/s | 1.47x | 84% |

A flat **~1.5x** for free. Almost the entire gain is captured at the very first draft
position, so the exact `n-max` barely matters; anything from 1 to 3 lands within noise.
The draft acceptance rate (how often the small model's guesses survive verification) is
high because the MTP head is trained alongside the model it is drafting for.

Generation speed is fundamentally **memory-bandwidth-bound**: at ~4B active parameters
per token, ~60 tok/s is the ceiling on this hardware, and no flag pushes past it.

## Choosing the model: why Qwen does not fit this Mac (yet)

The tempting next move is "use a bigger, smarter model," and on paper there is an obvious
candidate. Qwen's published benchmarks put Qwen 3.6 35B-A3B well ahead of Gemma 4 26B-A4B
on coding tasks including SWE-bench and LiveCodeBench, and it is an even leaner MoE (3B
active vs 4B). It looks like a strict upgrade. On this hardware it is not, and walking
through why is the whole point of the four questions above.

- **At Q4 it does not fit.** Qwen's Q4 weights are ~21 GB. Add the KV cache and macOS and
  you blow past 32 GiB, the machine swaps constantly, and it runs at **40 tok/s, slower
  than the 14 GB Gemma's 60** despite activating fewer parameters per token. A model that
  does not fit is slower than a smaller one that does, full stop.
- **At Q3 it fits but does not get faster.** Drop Qwen to Q3 (~16 GB) and the swapping
  stops (30% memory free), but generation only recovers to **~44 tok/s**, still well under
  Gemma. So swap was never the main cost: Qwen is simply heavier per token, and "fewer
  active params" does not capture the attention, routing, and embedding overhead. It is
  ~44 tok/s on this Mac at any quant that fits.
- **Q3 is the larger quality compromise.** Gemma runs at a Q4 *QAT* checkpoint
  ([quantization-aware trained](https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/)),
  which Google says minimizes quality loss compared with ordinary post-training
  quantization. To fit, Qwen has to run at the more aggressive Q3 quantization, and there
  is no public benchmark of the Q3 build used here. The full-model benchmark gap therefore
  does not tell you exactly how these two local builds compare.

Net: at the quants you can run on 32 GiB, Qwen is **slower (44 vs 60 tok/s), tighter on
memory, and requires more aggressive quantization to fit**, in exchange for a coding edge
that cannot be read directly from the full-model benchmarks. It is the better model in the
abstract and a great pick on a 48 GB+ machine, but **it is not the right fit for this
hardware right now.**

The model that wins on a 32 GiB Air is not the "best" one, it is the best one that fits
with headroom and reads few bytes per token. For this machine that is the 14 GB Gemma MoE.

The same logic rules out the dense option: Gemma 4 also ships a dense 31B, but "dense"
means all 31B parameters activate per token (versus 3-4B for the MoEs), so it crawls at
~10-15 tok/s, far too slow for an agent loop. Nor is a higher-precision Gemma quant an
obvious win: its Q4 checkpoint was trained to minimize quantization loss, so Q5/Q6/Q8 need
workload-specific evidence to justify their extra memory. Speculative decoding is designed
to preserve quality, and produced identical output in the greedy benchmark above. The
remaining quality levers are the samplers, the thinking budget, a good system prompt, and
enough context.

## Does it actually work as an agent?

The real test is not chat, it is the tool loop. Given a file with a deliberately broken
`factorial` function, the agent read the file, identified the bug, edited it
(`range(n)` to `range(1, n + 1)`), ran the script itself, and confirmed the output was
correct, all locally. The full read / edit / run loop works against the local model.

## Verdict

A year ago this would have been a fun toy. Today, on a mid-range laptop with no dedicated
GPU, it is a usable coding assistant: fast enough to stay in flow, smart enough to trust
with small real edits, and completely private and offline. It will not replace a frontier
model for hard problems, but for the long tail of routine edits, refactors, and "what
does this code do" questions, **local coding agents are feasible now.**

## Sources

- [How to setup a local coding agent on macOS – ikyle.me](https://ikyle.me/blog/2026/how-to-setup-a-local-coding-agent-on-macos)
- [llama.cpp MTP guide – Unsloth](https://unsloth.ai/docs/models/mtp#llama.cpp-mtp-guide)
- [Running local models is good now – Vicki Boykis](https://vickiboykis.com/2026/06/15/running-local-models-is-good-now/)
- [Gemma 4 – Google DeepMind](https://deepmind.google/models/gemma/gemma-4/)
- [Gemma 4 QAT models – Google](https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/)
- [Gemma 4 – How to Run Locally (Unsloth)](https://unsloth.ai/docs/models/gemma-4)
- [Gemma 4 likes high temperature for coding (HF discussion)](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/discussions/21)
- [pi: custom providers / models.json](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/models.md)
- [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [Qwen3.6-35B-A3B model card and benchmarks](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
