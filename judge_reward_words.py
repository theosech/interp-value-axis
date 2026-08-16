"""LLM judge #2: which words in a rewritten paragraph satisfy the criterion?

Why
---
The axis is constructed as (mean after - mean before) around the criterion-satisfying
word, but the corpus only ships those word labels for ONE paragraph per conversation
(the first post-discovery one, 380 instances). To compute the same within-attempt
contrast for every successful attempt we need the labels everywhere.

Syntactic criteria need no judge -- SYNTACTIC_TOKEN_CHECKERS finds the dash/colon/
emoji directly. Only the 1,484 semantic successes go to codex.

Judged text is the REWRITTEN PARAGRAPH only, never the <think> block: we are locating
where the criterion is satisfied in the output, which is independent of what the agent
believed it was doing.

Usage:  python judge_reward_words.py [--batch 15] [--workers 6] [--limit N]
Output: results/reward_words.jsonl  (resumable)
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "value-axis"))
sys.path.insert(0, str(ROOT / "value-axis" / "construction"))

from judge_rules import parse_response, run_codex  # noqa: E402  (reuses the codex plumbing)

OUT = ROOT / "results" / "reward_words.jsonl"
THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.S)

INSTRUCTIONS = """\
You are locating which words in a rewritten paragraph satisfy a hidden CRITERION.

For each item, output the exact substring(s) of the PARAGRAPH that satisfy the criterion.

Rules:
- Copy substrings VERBATIM from the paragraph, preserving case, spacing and punctuation.
- Usually there is exactly ONE. List more only if there are genuinely distinct instances.
- Pick the shortest span that satisfies the criterion (a word or short phrase, not a sentence).
- If nothing in the paragraph satisfies the criterion, output an empty list.

Answer from the context below alone. Do not run commands, read files, or use any tools.

Output exactly one line of JSON per item and nothing else:
{"id": <id>, "words": ["..."]}

ITEMS:
"""


def paragraph_text(body):
    """The rewritten paragraph: the assistant message minus its <think> block."""
    return THINK_RE.sub("", body or "").strip()


def load_items(limit=None):
    """Successful attempts on SEMANTIC criteria, keyed the same way as attempt_table."""
    from shared import load_conversations, load_reward_functions
    rf = {r["name"]: r for r in load_reward_functions()}
    from turns import attempt_records
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    def render(ms):
        try:
            return tok.apply_chat_template(ms, tokenize=False,
                                           add_generation_prompt=False, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=False)

    items = []
    for conv in load_conversations():
        if rf[conv["reward_fn"]]["type"] != "semantic":
            continue
        formatted = render(conv["full_messages"])
        recs = attempt_records(formatted, conv["discovery_paragraph"])
        for r in recs:
            if r["reward"] is not True:
                continue
            para = paragraph_text(formatted[r["assistant_start"]:r["assistant_end"]])
            if not para:
                continue
            items.append({
                "id": len(items),
                "conv_id": conv["conversation_id"], "reward_fn": conv["reward_fn"],
                "criterion": rf[conv["reward_fn"]]["description"],
                "paragraph": r["paragraph"],
                "attempt_in_paragraph": r["attempt_in_paragraph"],
                "text": para,
            })
    return items[:limit] if limit else items


def build_prompt(batch):
    out = [INSTRUCTIONS]
    for it in batch:
        out.append(f'[{it["id"]}] CRITERION: {it["criterion"]}\n'
                   f'      PARAGRAPH: {it["text"]}\n')
    return "\n".join(out)


def parse_words(stdout):
    text, err = None, None
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") == "item.completed":
            item = obj.get("item", {})
            if item.get("type") == "agent_message":
                text = item.get("text", "")
        elif obj.get("type") == "error":
            err = (obj.get("error") or {}).get("message") or obj.get("message")
    if text is None:
        raise RuntimeError(f"no agent_message; last error: {err}")
    rows = []
    for m in re.finditer(r'\{\s*"id"\s*:\s*(\d+)\s*,\s*"words"\s*:\s*\[[^\]]*\]\s*\}', text):
        try:
            rows.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=15)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = load_items(args.limit or None)
    OUT.parent.mkdir(exist_ok=True)
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(items)} semantic successes, {len(done)} labelled, {len(todo)} to do")
    if not todo:
        return

    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    print(f"{len(batches)} codex calls at {args.workers}-way concurrency", flush=True)
    by_id = {it["id"]: it for it in items}
    kept = dropped = 0

    def work(batch):
        try:
            return batch, parse_words(run_codex(build_prompt(batch))), None
        except Exception as e:                       # noqa: BLE001
            return batch, [], str(e)

    with OUT.open("a") as fh, ThreadPoolExecutor(max_workers=args.workers) as ex:
        for n, (batch, rows, err) in enumerate(ex.map(work, batches), 1):
            if err:
                print(f"  batch {n}/{len(batches)} FAILED: {err[:110]}", flush=True)
                continue
            for r in rows:
                it = by_id.get(r["id"])
                if it is None:
                    continue
                # Keep only spans that genuinely occur in the paragraph -- the judge
                # sometimes paraphrases, and a non-substring cannot be located.
                words = [w for w in (r.get("words") or [])
                         if isinstance(w, str) and w and w in it["text"]]
                dropped += len(r.get("words") or []) - len(words)
                kept += 1
                fh.write(json.dumps({k: it[k] for k in
                                     ("id", "conv_id", "reward_fn", "paragraph",
                                      "attempt_in_paragraph")} | {"words": words}) + "\n")
            fh.flush()
            print(f"  batch {n}/{len(batches)}: {len(rows)} labelled", flush=True)

    print(f"wrote {kept} rows to {OUT} ({dropped} non-substring spans dropped)")


if __name__ == "__main__":
    main()
