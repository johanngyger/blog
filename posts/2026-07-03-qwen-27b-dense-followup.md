# Follow-up: trying the dense Qwen 3.6 27B anyway

*3 July 2026*

[Last time](2026-06-21-local-coding-agents.md) I concluded that on a 32 GiB
MacBook Air M5, the model that wins is the one that fits with headroom and reads
few bytes per token, which ruled out Qwen's 35B-A3B MoE and, by the same
bandwidth math, every dense model. Then Quesma published
[Qwen 3.6 is awesome](https://quesma.com/blog/qwen-36-is-awesome/), making the
opposite trade on purpose: they run the **dense Qwen 3.6 27B** and argue

> I'd rather generate a third as much code, but of higher quality.

That is a coherent position, and the post comes with a one-liner setup
(`llama-server -hf unsloth/Qwen3.6-27B-MTP-GGUF:Q8_0 ...`). So I tried it on
the Air.

## The adaptation

Their Q8_0 weighs ~29 GB and was benchmarked on an M5 Max with 128 GB. On 32 GiB
the post itself points to 4-bit, so I used the **UD-Q4_K_XL** quant (17 GB) and
kept my usual server flags: MTP speculative decoding (the draft heads are
embedded in this GGUF), flash attention, q8_0 KV cache, and 32k context instead
of their 64k, because a dense model's KV cache is much heavier per token than
the MoEs'.

```sh
llama-server \
  -m Qwen3.6-27B-UD-Q4_K_XL.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  -ngl 999 -fa on -ctk q8_0 -ctv q8_0 -c 32768 \
  --reasoning on --reasoning-format deepseek \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0
```

It fits: ~16% memory free while serving, no meaningful swap. Everything works,
MTP runs at a 77% draft acceptance rate, and the thinking channel behaves.

## The numbers

| | Gemma 4 26B-A4B (MoE) | Qwen 3.6 27B (dense) |
|---|---|---|
| Generation | ~60 tok/s | **~9 tok/s** |
| Prefill (few-thousand-token prompt) | ~700 tok/s | **~130 tok/s** |

No surprises, and that is the point. Dense means every token reads all 27B
parameters, so at ~150 GB/s the ceiling is about 8 tok/s, and MTP only nudges
it. Quesma's 32 tok/s comes from the M5 Max's roughly 3.5x memory bandwidth,
not from anything the flags can replicate. The prefill number hurts more than
the generation number: an agent turn with a 20k-token context takes ~2.5 minutes
before the first output token.

## Verdict

The Quesma trade is "a third as much code, but of higher quality." On an M5 Max
that is a fair price. On this Air the same trade is closer to *a seventh* as
much code, plus minutes of prefill per agent turn, and that is no longer a
trade, it is a stall. The bandwidth arithmetic from the first post survives
contact with the tempting blog post: on 32 GiB and 150 GB/s, the small-active-set
MoE is still the right answer, and the dense 27B is a fine model waiting for a
better-fed machine.

## Sources

- [Qwen 3.6 is awesome – Quesma](https://quesma.com/blog/qwen-36-is-awesome/)
- [Qwen3.6-27B-MTP-GGUF – Unsloth](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF)
- [A local coding agent on a fanless laptop](2026-06-21-local-coding-agents.md)
