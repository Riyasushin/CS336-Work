"""PA3 Part 3: 投机解码 (Speculative Decoding)。

迁自 ``cse234-w25-PA/pa3/part3/PA3_Speculative_Decoding.ipynb``，按 src/ layout 组织。
保留原 scaffold 的类结构与方法签名（``initialize_target_model`` / ``initialize_draft_model``
/ ``generate_draft_tokens`` / ``verify_tokens_vectorized`` / ``speculative_decode``
/ ``benchmark``），并把 TODO 全部填实。

—— 算法（greedy 版，输出严格等同 ``do_sample=False`` 的 baseline）——

每轮迭代：
  1. draft_step:    Draft 模型 D 自回归 k 步，提议 d_0..d_{k-1}
  2. verify_step:   Target 模型 T 一次 forward([last_token, d_0..d_{k-1}])，得到 k+1 个
                    logits（最后一个 logit 落在 "bonus" 位置）
  3. accept:        逐位比对 argmax(logits[i]) vs d_i，找到第一个失配 j（全配则 j = k）
                    本轮提交 j 个接受 + 1 个 corrective/bonus = j + 1 个 token
  4. cache_trim:    把 T 与 D 的 KV cache 都截断到 ``committed_len + j``
                    （corrective/bonus 不进 cache，下一轮 draft 一开始喂它）

—— KV cache 约定 ——

  * cache 长度 = committed_len - 1（即不含最后一个已提交 token）
  * ``self._last_token_id`` 持有"最后一个已提交 token"
  * 下一次 verify 把它和 k 个 draft token 一起喂给 T，T 得到 k+1 个新位置的 logit，
    刚好覆盖「k 个 verify 位 + 1 个 bonus 位」。这样既不重算 prefix，也不丢 bonus。

—— 为什么 greedy speculative 输出 == greedy baseline ——

  * verify 阶段对每个 draft 位置都按 T 的 argmax 比对；只接受 "T 也会出的 token"
  * 拒绝点用 T 自己的 argmax 当 corrective ⇒ 等价于 T 在该位置自己跑一步
  * 全接受时 bonus = T 在新位置的 argmax ⇒ 也是 T 该位置自己会出的 token
  * 故任何接受/拒绝结果，最终序列都等于 T 单独跑 greedy generate(do_sample=False)
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


class SpeculativeDecoder:
    """Greedy speculative decoder（原 PA3 scaffold 的实装版本）。"""

    def __init__(self, target_model_name: str, draft_model_name: str, device: str = "cuda"):
        """初始化 target / draft 模型，并校验词表兼容。"""
        self.device = device
        self.target_model, self.target_tokenizer = self.initialize_target_model(target_model_name)
        self.draft_model, self.draft_tokenizer = self.initialize_draft_model(draft_model_name)

        # speculative decoding 要求两个模型 token id 空间一致
        # （否则 draft 的 token id 喂给 target 没有意义）
        assert self.target_tokenizer.get_vocab() == self.draft_tokenizer.get_vocab(), \
            "target / draft tokenizer 词表不一致：speculative decoding 需要相同 vocab"

        # ---- per-decode session 状态（在 speculative_decode 起始处重置） ----
        self._target_cache: Optional[DynamicCache] = None
        self._draft_cache: Optional[DynamicCache] = None
        self._last_token_id: Optional[torch.Tensor] = None       # 形状 [1, 1]
        self._last_corrective_id: Optional[torch.Tensor] = None  # verify 结束后存放
        self._committed_len: int = 0  # 已提交 token 数（含 prompt）

    # ============================================================
    # 模型初始化
    # ============================================================
    def initialize_target_model(self, model_name: str):
        """Target 模型：fp16 + sdpa attention + eval 模式。"""
        print(f"Loading target model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 1. pad token：很多 LM (gpt2/llama) 默认没有 pad，用 eos 顶替
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 2. 加载模型：fp16 让 wall-clock 跑得动；sdpa 是 PyTorch 内置 fused attention
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        model.to(self.device)
        model.eval()  # 关 dropout
        # 3. 同步 pad_token_id 到 model.config（generate() 默认要用）
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def initialize_draft_model(self, model_name: str):
        """Draft 模型：与 target 同样的 fp16 + sdpa；体量更小所以更快。"""
        print(f"Loading draft model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        model.to(self.device)
        model.eval()
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    # ============================================================
    # 内部：prefill + cache 截断
    # ============================================================
    @torch.inference_mode()
    def _prefill(self, input_ids: torch.Tensor) -> None:
        """初始化 T / D 的 KV cache，使其覆盖 input_ids[:-1]。

        约定：cache 持有 ``committed_len - 1`` 个位置的 K/V，最后一个已提交 token
        放在 ``self._last_token_id`` 里。下一次 verify 会把它和 k 个 draft 一起喂给 T。
        """
        prefix = input_ids[:, :-1]  # 不含最后一个 prompt token
        self._target_cache = DynamicCache()
        self._draft_cache = DynamicCache()
        if prefix.shape[1] > 0:
            t_out = self.target_model(
                input_ids=prefix, past_key_values=self._target_cache, use_cache=True
            )
            self._target_cache = t_out.past_key_values
            d_out = self.draft_model(
                input_ids=prefix, past_key_values=self._draft_cache, use_cache=True
            )
            self._draft_cache = d_out.past_key_values
        self._last_token_id = input_ids[:, -1:].clone()
        self._committed_len = input_ids.shape[1]

    def _trim_caches_after_accept(self, j: int) -> None:
        """接受 j 个 draft 后，把 T/D 两边 cache 都截到 ``committed_len + j``。

        截断前：
          target cache 长度 = committed_len - 1 + (k+1) = committed_len + k
          draft  cache 长度 = committed_len - 1 + k     = committed_len + k - 1
        截断后：
          两边都覆盖 [0, committed_len + j)，对应「prompt + 已接受 draft」全部位置。
          corrective/bonus token 不进 cache，由 ``_last_token_id`` 持有，下轮喂回去。
        """
        target_keep = self._committed_len + j
        draft_keep = self._committed_len + j
        # DynamicCache.crop(max_length) 会原地把所有 layer 的 K/V 截到 max_length
        self._target_cache.crop(target_keep)
        self._draft_cache.crop(draft_keep)

    # ============================================================
    # PA3 三个核心 step
    # ============================================================
    @torch.inference_mode()
    def generate_draft_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_speculative_tokens: int = 10,
    ) -> torch.Tensor:
        """用 draft 模型自回归推 k 个 token；每步只喂 1 个 token，复用 KV cache。

        签名沿用 PA3 scaffold：``input_ids`` / ``attention_mask`` 仅作为上下文标记
        （表明"截至此刻"的提交状态），真实推理由内部 ``_draft_cache`` + ``_last_token_id``
        驱动。这样可以避免每轮重算整段 prefix。
        """
        del input_ids, attention_mask  # 仅签名兼容
        k = num_speculative_tokens

        # 起点：last_committed_token 喂给 D，拿位置 committed_len 的 logit
        # 之后 k-1 步，每步把上一步采的 draft token 喂回去
        feed = self._last_token_id  # [1, 1]
        drafts = []
        for _ in range(k):
            out = self.draft_model(
                input_ids=feed, past_key_values=self._draft_cache, use_cache=True
            )
            self._draft_cache = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
            drafts.append(next_id)
            feed = next_id

        # 关键：把最后一个 draft 也喂回 D 一次（仅为了同步 cache，丢弃 logits）。
        # 否则 D 的 cache 比 T 短 1（T 在 verify 时把 [last, d_0..d_{k-1}] 全喂了），
        # 当 j == k 全接受时，下一轮 D 喂回 corrective/bonus 的位置就会错位。
        out = self.draft_model(
            input_ids=feed, past_key_values=self._draft_cache, use_cache=True
        )
        self._draft_cache = out.past_key_values
        return torch.cat(drafts, dim=1)  # [1, k]

    @torch.inference_mode()
    def verify_tokens_vectorized(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[List[int], int]:
        """一次 forward 验证 k 个 draft token。

        Returns:
            accepted_tokens: 被接受的 draft token id 列表（长度 = ``j``）
            accepted_position: 第一个被拒绝的位置 ``j``（全接受则 ``j = k``）

        Side effect:
            * ``self._last_corrective_id``: T 在第 j 个位置的 argmax，作为 corrective
              （或 bonus，当 j == k）；shape [1, 1]
            * ``self._target_cache`` 长度从 ``committed_len - 1`` 涨到 ``committed_len + k``
        """
        del input_ids, attention_mask
        k = draft_tokens.shape[1]

        # 一次喂 [last_committed, d_0, ..., d_{k-1}]，共 k+1 个 token
        # logits 出来 k+1 个位置：
        #   logits[:, 0, :] 预测 d_0 应该是什么
        #   logits[:, i, :] 预测 d_i 应该是什么 (i in 0..k-1)
        #   logits[:, k, :] 预测 bonus token（当 k 个全接受时使用）
        feed = torch.cat([self._last_token_id, draft_tokens], dim=1)  # [1, k+1]
        out = self.target_model(
            input_ids=feed, past_key_values=self._target_cache, use_cache=True
        )
        self._target_cache = out.past_key_values
        logits = out.logits  # [1, k+1, V]
        preds = logits.argmax(dim=-1)  # [1, k+1]

        # preds[0, :k] 与 draft_tokens 逐位比对
        match = preds[:, :k].eq(draft_tokens)  # [1, k] bool
        if bool(match.all()):
            j = k  # 全部接受
        else:
            # 第一个 False 的位置：对 (1 - match.int()) 取 argmax 即"第一个 1"
            j = int(match.int().argmin(dim=1).item())

        accepted = draft_tokens[0, :j].tolist()
        # corrective (j < k) 或 bonus (j == k) 都正好是 preds[:, j]
        # 因为 logits[j] 是 T 在第 j 个新位置的预测；而 j 范围是 [0, k]，刚好和 logits 维度对齐
        self._last_corrective_id = preds[:, j : j + 1].clone()  # [1, 1]
        return accepted, j

    # ============================================================
    # 主循环
    # ============================================================
    @torch.inference_mode()
    def speculative_decode(
        self,
        prompt: str,
        max_tokens: int = 100,
        num_speculative_tokens: int = 5,
    ) -> str:
        """投机解码主入口；返回 prompt + 生成内容的字符串。"""
        inputs = self.target_tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        prompt_length = input_ids.shape[1]

        eos_id = self.target_tokenizer.eos_token_id

        # 初始化 cache：cache 覆盖 prompt[:-1]，_last_token_id = prompt[-1]
        self._prefill(input_ids)

        total_tokens_generated = prompt_length
        total_draft_tokens_proposed = 0
        total_draft_tokens_accepted = 0
        start_time = time.time()

        generated = input_ids  # 整段序列；用于结束后 decode
        stop = False

        while (generated.shape[1] - prompt_length) < max_tokens and not stop:
            remaining = max_tokens - (generated.shape[1] - prompt_length)
            # 最后一轮可能比 num_speculative_tokens 小；这样不会超 max_tokens 太多
            k = min(num_speculative_tokens, max(1, remaining))

            # 1) draft：D 推 k 个候选
            drafts = self.generate_draft_tokens(generated, attention_mask, k)
            # 2) verify：T 一次 forward 验完
            accepted, j = self.verify_tokens_vectorized(generated, drafts, attention_mask)

            total_draft_tokens_proposed += k
            total_draft_tokens_accepted += j

            corrective = self._last_corrective_id  # [1, 1]
            # 本轮提交的新 token：j 个接受 + 1 个 corrective/bonus
            new_tokens = torch.cat([drafts[:, :j], corrective], dim=1)  # [1, j+1]
            generated = torch.cat([generated, new_tokens], dim=1)

            # 3) cache 截断：让 T/D cache 都覆盖到 committed + 接受位
            self._trim_caches_after_accept(j)
            # corrective/bonus 不在 cache 中；下轮 draft 一开始喂它给 D
            self._last_token_id = corrective
            self._committed_len = generated.shape[1]

            # EOS 检查：只要本轮新 token 里出现 eos，就停
            if eos_id is not None and bool((new_tokens == eos_id).any()):
                stop = True

        total_tokens_generated = generated.shape[1]
        elapsed_time = time.time() - start_time
        acceptance_rate = (
            total_draft_tokens_accepted / total_draft_tokens_proposed
            if total_draft_tokens_proposed > 0 else 0.0
        )

        # 截断到 max_tokens 长度（若最后一轮 j+1 多出几个）
        final = generated[:, : prompt_length + max_tokens]

        new_token_count = final.shape[1] - prompt_length
        print(f"Generated {new_token_count} tokens in {elapsed_time:.2f} seconds")
        print(f"Tokens per second: {new_token_count / elapsed_time:.2f}")
        print(f"Draft token acceptance rate: {acceptance_rate:.2%}")

        return self.target_tokenizer.decode(final[0], skip_special_tokens=True)

    # ============================================================
    # benchmark
    # ============================================================
    def benchmark(
        self,
        prompt: str,
        max_tokens: int = 100,
        num_runs: int = 3,
        compare_baseline: bool = True,
    ) -> Dict:
        """与 baseline (target.generate) 比对 wall-clock 与 tokens/s。"""
        results = {
            "speculative": {"times": [], "tokens_per_second": []},
            "baseline": {"times": [], "tokens_per_second": []} if compare_baseline else None,
        }

        # speculative
        for _ in range(num_runs):
            start_time = time.time()
            output = self.speculative_decode(prompt, max_tokens=max_tokens)
            elapsed = time.time() - start_time
            prompt_len = len(self.target_tokenizer(prompt)["input_ids"])
            output_tokens = max(1, len(self.target_tokenizer.encode(output)) - prompt_len)
            tps = output_tokens / elapsed
            results["speculative"]["times"].append(elapsed)
            results["speculative"]["tokens_per_second"].append(tps)

        # baseline = target greedy generate
        if compare_baseline:
            for _ in range(num_runs):
                inputs = self.target_tokenizer(prompt, return_tensors="pt", padding=True)
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)
                start_time = time.time()
                with torch.inference_mode():
                    output_ids = self.target_model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_length=input_ids.shape[1] + max_tokens,
                        do_sample=False,
                        pad_token_id=self.target_tokenizer.pad_token_id,
                    )
                elapsed = time.time() - start_time
                output_tokens = output_ids.shape[1] - input_ids.shape[1]
                tps = output_tokens / elapsed
                results["baseline"]["times"].append(elapsed)
                results["baseline"]["tokens_per_second"].append(tps)

        for method in list(results.keys()):
            if results[method] is None:
                continue
            avg_time = sum(results[method]["times"]) / num_runs
            avg_tps = sum(results[method]["tokens_per_second"]) / num_runs
            results[method]["avg_time"] = avg_time
            results[method]["avg_tokens_per_second"] = avg_tps

        if compare_baseline:
            results["speedup"] = (
                results["baseline"]["avg_time"] / results["speculative"]["avg_time"]
            )
            results["latency_reduction"] = (
                1 - results["speculative"]["avg_time"] / results["baseline"]["avg_time"]
            ) * 100
        return results


# ============================================================
# CLI bench（原 ipynb 的 Test 单元改写：可指定 target/draft + 直接出报表）
# ============================================================
def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Speculative decoding benchmark (PA3 Part 3)")
    parser.add_argument("--target", type=str, default="gpt2-medium",
                        help="HF model id for target (larger)")
    parser.add_argument("--draft", type=str, default="gpt2",
                        help="HF model id for draft (smaller)")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cpu；默认自动检测")
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--k", type=int, default=4,
                        help="num_speculative_tokens")
    parser.add_argument("--num_runs", type=int, default=3)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    decoder = SpeculativeDecoder(args.target, args.draft, device=device)

    test_prompts = [
        "The future of Artificial Intelligence is",
        "Write a short story about a robot learning to feel emotions:",
        "Write the lyrics to the song 'Happy Birthday'.",
    ]

    # 整体 warmup（CUDA init + autotune 不计入第一条 prompt 的均值）
    if device == "cuda":
        _ = decoder.speculative_decode("Hello", max_tokens=8, num_speculative_tokens=args.k)

    summary = []
    for i, prompt in enumerate(test_prompts):
        print(f"\n=== Prompt {i + 1}: {prompt!r}")
        r = decoder.benchmark(
            prompt=prompt,
            max_tokens=args.max_tokens,
            num_runs=args.num_runs,
            compare_baseline=True,
        )
        spec_t = r["speculative"]["avg_time"]
        spec_tps = r["speculative"]["avg_tokens_per_second"]
        base_t = r["baseline"]["avg_time"]
        base_tps = r["baseline"]["avg_tokens_per_second"]
        summary.append((prompt, spec_t, spec_tps, base_t, base_tps, r["speedup"]))
        print(f"  speculative: {spec_t:.3f}s  ({spec_tps:.2f} tok/s)")
        print(f"  baseline   : {base_t:.3f}s  ({base_tps:.2f} tok/s)")
        print(f"  speedup    : {r['speedup']:.2f}×   latency-reduction: {r['latency_reduction']:.1f}%")

    print("\n" + "=" * 70)
    print(f"target = {args.target} | draft = {args.draft} | k = {args.k} | device = {device}")
    print(f"{'prompt':<60s}  {'spec':>6s}  {'base':>6s}  {'speedup':>7s}")
    for p, st, _, bt, _, sp in summary:
        print(f"{p[:60]:<60s}  {st:6.2f}  {bt:6.2f}  {sp:6.2f}×")


if __name__ == "__main__":
    _main()
