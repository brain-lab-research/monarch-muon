import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import pyarrow.parquet as parquet
from transformers import AutoTokenizer


def encode_conversation(tokenizer, messages, max_length):
    if not messages or any(m["role"] not in ("system", "user", "assistant")
                           or not isinstance(m["content"], str) for m in messages):
        return None
    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=False)
    spans = []
    for i, message in enumerate(messages):
        if message["role"] != "assistant" or i == 0:
            continue
        before = tokenizer.apply_chat_template(messages[:i], tokenize=False,
                                                add_generation_prompt=True)
        through = tokenizer.apply_chat_template(messages[:i + 1], tokenize=False,
                                                 add_generation_prompt=False)
        if not through.startswith(before) or not rendered.startswith(through):
            raise ValueError("Chat template is not prefix-compatible")
        # Include the assistant's im_end token, excluding the trailing newline.
        spans.append((len(before), len(through.rstrip("\n"))))
    tokens = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    labels = [token if end > start and any(start >= a and end <= b for a, b in spans)
              else -100 for token, (start, end) in zip(tokens["input_ids"], tokens["offset_mapping"])]
    full_supervised = sum(t != -100 for t in labels[1:])
    input_ids, labels = tokens["input_ids"][:max_length], labels[:max_length]
    supervised = sum(t != -100 for t in labels[1:])
    if not supervised:
        return None
    return {"input_ids": input_ids, "labels": labels,
            "original_length": len(tokens["input_ids"]),
            "supervised_tokens": supervised,
            "dropped_supervised_tokens": full_supervised - supervised}


def read_split(path):
    return [json.loads(line) for line in Path(path).open()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="sft_config.json")
    parser.add_argument("--model-dir", required=True,
                        help="Local Qwen2.5-7B directory (only its tokenizer and chat template are used).")
    parser.add_argument("--tulu-dir", required=True,
                        help="Local Tulu-3 SFT mixture directory containing data/train-0000{0..5}-of-00006.parquet.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration-examples", type=int, default=0)
    parser.add_argument("--calibration-validation-examples", type=int, default=0)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("Prepared data already exists; use its recorded manifest")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    rng = random.Random(config["seed"])
    wanted = config["train_examples"] + config["validation_examples"] + config["test_examples"]
    pool_size = wanted * 3
    wanted += args.calibration_examples + args.calibration_validation_examples
    pool, seen = [], 0
    for shard in range(6):
        path = Path(args.tulu_dir) / f"data/train-{shard:05d}-of-00006.parquet"
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=256):
            for row in batch.to_pylist():
                seen += 1
                item = {"messages": row["messages"], "source_row": seen - 1,
                        "source": row.get("source", "unknown")}
                if len(pool) < pool_size:
                    pool.append(item)
                else:
                    index = rng.randrange(seen)
                    if index < pool_size:
                        pool[index] = item
    rng.shuffle(pool)
    rows, hashes = [], set()
    skipped, duplicated = 0, 0
    for item in pool:
        message_hash = hashlib.sha256(json.dumps(item["messages"], ensure_ascii=False,
                                                sort_keys=True).encode()).hexdigest()
        if message_hash in hashes:
            duplicated += 1
            continue
        encoded = encode_conversation(tokenizer, item["messages"], config["sequence_length"])
        if encoded is None:
            skipped += 1
            continue
        hashes.add(message_hash)
        encoded.update(source_row=item["source_row"], source=item["source"],
                       conversation_sha256=message_hash)
        rows.append(encoded)
        if len(rows) == wanted:
            break
    if len(rows) != wanted:
        raise RuntimeError(f"Only {len(rows)} usable examples in the fixed candidate sample")
    manifest = {"config": config, "source_rows": seen, "skipped_no_target_or_schema": skipped,
                "skipped_duplicate": duplicated, "splits": {},
                "template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()}
    start = 0
    for name, count in [("train", config["train_examples"]),
                        ("validation", config["validation_examples"]),
                        ("test", config["test_examples"]),
                        ("calibration", args.calibration_examples),
                        ("calibration_validation", args.calibration_validation_examples)]:
        if count == 0:
            continue
        split = rows[start:start + count]
        start += count
        body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in split)
        (output / f"{name}.jsonl").write_text(body)
        manifest["splits"][name] = {
            "examples": len(split), "sha256": hashlib.sha256(body.encode()).hexdigest(),
            "supervised_tokens": sum(r["supervised_tokens"] for r in split),
            "truncated_examples": sum(r["original_length"] > config["sequence_length"] for r in split),
            "dropped_supervised_tokens": sum(r["dropped_supervised_tokens"] for r in split),
            "sources": dict(Counter(r["source"] for r in split)),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
