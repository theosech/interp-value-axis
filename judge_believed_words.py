"""LLM judge #3: where did the agent implement ITS OWN stated hypothesis?

Why
---
`judge_reward_words.py` locates the word that ACTUALLY satisfies the hidden criterion.
That only exists for successful attempts, and for a lucky (`RW`) attempt it is not where
the agent thought it was succeeding.

This judge locates the BELIEVED reward word: the span implementing whatever hypothesis the
reasoning states, right or wrong. It exists for every attempt, which

  1. extends the within-attempt (after - before) contrast to the 2,383 `FW` and 62 `FC`
     attempts -- the balanced comparison the actual-word divider cannot reach; and
  2. for `RW` attempts, gives TWO positions in the same paragraph -- where the model
     believed it succeeded and where it actually did. Those come apart only in `RW`, and
     which one the axis rises at discriminates a confidence/expected-value reading from a
     "this text exhibits the feature" reading.

Judged from the REASONING plus the PARAGRAPH. Correctness of the hypothesis is irrelevant
here -- we want where the agent placed its bet.

Not every hypothesis is localizable ("use passive voice", "add sensory detail"). Those are
returned as diffuse rather than forced onto a span, and coverage is reported.

Usage:  python judge_believed_words.py [--batch 12] [--workers 6] [--limit N]
Output: results/believed_words.jsonl  (resumable)
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

from judge_reward_words import paragraph_text  # noqa: E402
from judge_rules import run_codex  # noqa: E402

OUT = ROOT / "results" / "believed_words.jsonl"
THINK_RE = re.compile(r"<think(?:ing)?>(.*?)</think(?:ing)?>", re.S)

INSTRUCTIONS = """\
You are locating where an agent implemented ITS OWN stated hypothesis.

Each item gives the agent's private REASONING (which states a hypothesis about a hidden rule
it is trying to satisfy) and the PARAGRAPH it then wrote.

Output the exact substring(s) of the PARAGRAPH that implement the hypothesis stated in the
REASONING -- the place the agent believes it satisfied the rule.

Judge by the STATED hypothesis, not by whether that hypothesis is correct. If the reasoning
says "the rule is adding a rhetorical question", point at the rhetorical question, even if
that is not really the rule.

Rules:
- Copy substrings VERBATIM from the paragraph, preserving case, spacing and punctuation.
- Pick the SHORTEST span that implements the hypothesis. Usually one word or phrase.
- If the hypothesis is not localizable to a short span -- e.g. "use passive voice",
  "add sensory detail", "restructure the sentences", "make it more concise" -- output an
  empty list and set "diffuse": true.
- If the reasoning states no hypothesis at all, output an empty list and "diffuse": true.

Answer from the context below alone. Do not run commands, read files, or use any tools.

Also rate how CONFIDENT the reasoning sounds about its hypothesis, 1-5:

  1 = lost. No hypothesis, or explicitly out of ideas ("nothing is working", "I don't know
      what the rule could be").
  2 = tentative guess. Floats a possibility with no support ("the criterion might be about
      X", "maybe X", "I'll try X to test this").
  3 = working hypothesis with some evidence. Cites feedback but hedges ("the feedback
      pattern suggests X", "X seems to be the criterion").
  4 = confident. Asserts the rule and cites consistent evidence ("the pattern is clear now",
      "I'm fairly confident the criterion is X").
  5 = certain. States the rule as settled fact ("the criterion is definitely X", "I've
      confirmed the criterion is X", "I get +1 every time I include X").

Rate the EXPRESSED confidence, independently of whether the hypothesis is correct.

Output exactly one line of JSON per item and nothing else:
{"id": <id>, "words": ["..."], "diffuse": true, "confidence": 3}

ITEMS:
"""


def load_items(limit=None):
    from shared import load_conversations, load_reward_functions
    from turns import attempt_records
    from transformers import AutoTokenizer

    rf = {r["name"]: r for r in load_reward_functions()}
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    def render(ms):
        try:
            return tok.apply_chat_template(ms, tokenize=False,
                                           add_generation_prompt=False, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=False)

    items = []
    for conv in load_conversations():
        formatted = render(conv["full_messages"])
        for r in attempt_records(formatted, conv["discovery_paragraph"]):
            body = formatted[r["assistant_start"]:r["assistant_end"]]
            m = THINK_RE.search(body)
            think = m.group(1).strip() if m else ""
            para = paragraph_text(body)
            if not think or not para:
                continue
            items.append({
                "id": len(items),
                "conv_id": conv["conversation_id"], "reward_fn": conv["reward_fn"],
                "type": rf[conv["reward_fn"]]["type"],
                "paragraph": r["paragraph"],
                "attempt_in_paragraph": r["attempt_in_paragraph"],
                "reward": r["reward"],
                "think": think, "text": para,
            })
    return items[:limit] if limit else items


def build_prompt(batch):
    out = [INSTRUCTIONS]
    for it in batch:
        out.append(f'[{it["id"]}] REASONING: {it["think"]}\n'
                   f'      PARAGRAPH: {it["text"]}\n')
    return "\n".join(out)


def parse_believed(stdout):
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
    for m in re.finditer(r'\{\s*"id"\s*:\s*\d+\s*,[^{}]*\}', text):
        try:
            rows.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            pass
    return [r for r in rows if "id" in r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=12)
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
    print(f"{len(items)} attempts, {len(done)} labelled, {len(todo)} to do")
    if not todo:
        return

    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    print(f"{len(batches)} codex calls at {args.workers}-way concurrency", flush=True)
    by_id = {it["id"]: it for it in items}
    kept = dropped = diffuse = no_conf = 0

    def work(batch):
        try:
            return batch, parse_believed(run_codex(build_prompt(batch))), None
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
                words = [w for w in (r.get("words") or [])
                         if isinstance(w, str) and w and w in it["text"]]
                dropped += len(r.get("words") or []) - len(words)
                diffuse += bool(r.get("diffuse")) or not words
                conf = r.get("confidence")
                conf = int(conf) if isinstance(conf, (int, float)) and 1 <= conf <= 5 else None
                kept += 1
                no_conf += conf is None
                fh.write(json.dumps({k: it[k] for k in
                                     ("id", "conv_id", "reward_fn", "type", "paragraph",
                                      "attempt_in_paragraph", "reward")}
                                    | {"believed_words": words,
                                       "diffuse": bool(r.get("diffuse")) or not words,
                                       "confidence": conf}) + "\n")
            fh.flush()
            print(f"  batch {n}/{len(batches)}: {len(rows)} labelled", flush=True)

    print(f"wrote {kept} rows ({diffuse} diffuse, {dropped} non-substring spans dropped, "
          f"{no_conf} missing a confidence rating)")


if __name__ == "__main__":
    main()
