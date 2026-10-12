"""lm_eval wrapper for MMLU and GSM8K on the (patched) model in memory.

Benchmarks run through the EleutherAI lm-evaluation-harness, with its HFLM wrapper around the already patched model:
MMLU (acc) and GSM8K (exact_match, strict + flexible), every metric with the harness's own stderr.
run_joint_blocks.py calls `run_lm_eval` once per (configuration, benchmark).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from model_loading import final_hidden


def flatten_lm_eval_results(results: dict) -> list[dict]:
    """lm_eval `results['results']` -> rows (task, metric, filter, value, stderr, n).
    Keys look like 'acc,none' / 'acc_stderr,none' / 'exact_match,strict-match'; each value key is
    paired with its '<metric>_stderr,<filter>' key when present. `n` comes from results['n-samples']
    when available."""
    rows = []
    n_samples = results.get("n-samples", {})
    for task, metrics in results["results"].items():
        for key, value in metrics.items():
            if key == "alias" or "_stderr" in key or not isinstance(value, (int, float)):
                continue
            metric, _, flt = key.partition(",")
            se = metrics.get(f"{metric}_stderr,{flt}", float("nan"))
            rows.append(dict(task=task, metric=metric, filter=flt, value=float(value),
                             stderr=float(se) if isinstance(se, (int, float)) else float("nan"),
                             n=(n_samples.get(task, {}) or {}).get("effective", "")))
    return rows


_TASK_MANAGER = None        # lm_eval indexes ~14k task YAMLs when a TaskManager is built: once per process, not per call
_LAST_POSITION_HFLM = None


def last_position_hflm():
    """HFLM whose loglikelihood (MMLU and every other multiple-choice task) applies the LM head only at the continuation
    positions. The stock path takes the logits of EVERY position over the whole vocabulary -- Moonlight's remote forward
    even casts them to fp32 -- and log-softmaxes all of it: batch x length x 163840 x 4 bytes, ~30 GiB at batch 16 on a
    5-shot MMLU prompt, which is what held the benchmark at batch 2-4 (thousands of small forwards per replacement).
    Same requests, same truncation, right padding and one forward per distinct input as lm_eval (the four choices of a
    question share it); the result per request is lm_eval's (sum of continuation log-probs, greedy match)."""
    global _LAST_POSITION_HFLM
    if _LAST_POSITION_HFLM is not None:
        return _LAST_POSITION_HFLM
    import torch.nn.functional as F
    from lm_eval.models.huggingface import HFLM
    from tqdm import tqdm

    class LastPositionHFLM(HFLM):
        @torch.no_grad()
        def _loglikelihood_tokens(self, requests, disable_tqdm=False, override_bs=None):
            bs = override_bs or self.batch_size
            if self.backend != "causal" or not isinstance(bs, int):
                return super()._loglikelihood_tokens(requests, disable_tqdm, override_bs)
            head = self.model.get_output_embeddings()
            groups: dict[tuple, list] = {}                   # model input -> [(continuation, request index, request string)]
            for i, (req_str, ctx, cont) in enumerate(requests):
                inp = tuple((ctx + cont)[-(self.max_length + 1):][:-1])
                groups.setdefault(inp, []).append((list(cont), i, req_str))
            order = sorted(groups, key=len, reverse=True)
            res = [None] * len(requests)
            for b0 in tqdm(range(0, len(order), bs), disable=disable_tqdm or self.rank != 0, desc="loglikelihood (last positions)"):
                batch = order[b0:b0 + bs]
                ids = torch.zeros(len(batch), len(batch[0]), dtype=torch.long)
                for r, inp in enumerate(batch):
                    ids[r, :len(inp)] = torch.tensor(inp)
                hidden = final_hidden(self.model, ids)     # right padding: a causal model's real positions do not see it
                scored, sums, greedy = [], [], []
                for r, inp in enumerate(batch):
                    conts = groups[inp]
                    n_max = max(len(c) for c, _, _ in conts)
                    logp = F.log_softmax(head(hidden[r, len(inp) - n_max:len(inp)]).float(), dim=-1)
                    for cont, i, req_str in conts:
                        tgt = torch.tensor(cont, device=logp.device)
                        lp = logp[n_max - len(cont):]
                        sums.append(lp.gather(-1, tgt[:, None]).sum())
                        greedy.append((lp.argmax(-1) == tgt).all())
                        scored.append((i, req_str))
                for (i, req_str), ll, g in zip(scored, torch.stack(sums).tolist(), torch.stack(greedy).tolist()):
                    res[i] = (ll, bool(g))
                    if req_str is not None:
                        self.cache_hook.add_partial("loglikelihood", req_str, res[i])
            return res

    _LAST_POSITION_HFLM = LastPositionHFLM
    return LastPositionHFLM


def run_lm_eval(model, tokenizer, benchmark, fewshot, args) -> list[dict]:
    """One benchmark through lm_eval on the (patched) model. `args`: batch_size, limit and, optionally, samples_path
    (a JSONL file that gets one line per document: its metric values and, for generation, the response length)."""
    global _TASK_MANAGER
    try:
        import lm_eval
        from lm_eval.tasks import TaskManager
    except ImportError as exc:
        # (the message of the ImportError matters: lm_eval may be installed but fail to import against this transformers)
        raise SystemExit(f"backend lm_eval cannot be imported ({type(exc).__name__}: {exc}). Needs the lm-evaluation-harness "
                         "(pip install lm_eval) in a version that works with the installed transformers.") from exc
    if _TASK_MANAGER is None:
        _TASK_MANAGER = TaskManager()
    lm = last_position_hflm()(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
    samples_path = getattr(args, "samples_path", None)
    kwargs = dict(model=lm, tasks=[benchmark], limit=args.limit, log_samples=samples_path is not None,
                  task_manager=_TASK_MANAGER)
    if benchmark in fewshot:
        kwargs["num_fewshot"] = fewshot[benchmark]
    results = lm_eval.simple_evaluate(**kwargs)
    if samples_path is not None:
        write_samples(results, Path(samples_path))
    return flatten_lm_eval_results(results)


def write_samples(results: dict, path: Path) -> None:
    """Per-document metric values (paired tests against the dense run) and generated lengths, without the prompts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for task, samples in results.get("samples", {}).items():
            for s in samples:
                row = {"task": task, "doc_id": s["doc_id"], **{m: s[m] for m in s.get("metrics", []) if m in s}}
                resp = (s.get("resps") or [[None]])[0][0]
                if isinstance(resp, str):
                    row["resp_chars"] = len(resp)
                f.write(json.dumps(row) + "\n")
