"""LLM judge: does an attempt's <think> block state the CORRECT hidden criterion?

Why this exists
---------------
The corpus ships a `discovery_paragraph` field, but it does not mark where the agent
first states the correct rule. On syntactic criteria, 38% of attempts *before* that
paragraph already name the rule. So the positional split (lucky = before the label)
is contaminated and has to be replaced by a semantic one.

What is judged
--------------
The `<think>` block ONLY -- never the rewritten paragraph. The paragraph is the
*behaviour* (did it satisfy the criterion), which we already have as `reward`. The
whole point of the design is that a "lucky" hit is a WRONG hypothesis that happened
to satisfy the criterion; showing the judge the output would leak the behavioural
signal into the hypothesis label and collapse that distinction.

Backend: `codex exec --json`. Batched hard -- every call pays ~17k input tokens of
codex harness scaffolding regardless of prompt size, so one call per item would be
absurd. See the codex-exec-json-backend skill.

Usage:  python judge_rules.py [--batch 30] [--workers 6] [--limit N]
Output: results/rule_labels.jsonl  (one row per attempt, resumable)
"""

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "value-axis"))
sys.path.insert(0, str(ROOT / "value-axis" / "construction"))

OUT = ROOT / "results" / "rule_labels.jsonl"
WORKDIR = ROOT / ".codex_scratch"          # dedicated empty dir so codex cannot wander

THINK_RE = re.compile(r"<think(?:ing)?>(.*?)</think(?:ing)?>", re.S)
# Generator scaffolding that leaked into the transcripts ("stick to my wrong
# hypothesis"). 4.4% of blocks. Flagged, not judged -- so they can be excluded.
LEAK_RE = re.compile(r"wrong hypothesis|stick to my wrong|my wrong|pretend|incorrect hypothesis", re.I)

INSTRUCTIONS = """\
You are labelling private reasoning from a rule-guessing game. An agent must rewrite \
paragraphs so they satisfy a hidden CRITERION, and it writes its reasoning before each \
attempt.

For each item, decide whether the REASONING states the true CRITERION as its current \
working hypothesis.

Labels:
  correct - the reasoning names the true criterion as what it now believes/will do, \
even tentatively ("the criterion is probably X", "I need to include X")
  wrong   - the reasoning names a different hypothesis
  unclear - no hypothesis is named, or it is too vague to tell (e.g. "I'll try the \
approach that worked")

Judge ONLY the reasoning text. Do not reward the agent for describing an approach that \
would incidentally satisfy the criterion -- it must actually name the criterion.

Answer from the context below alone. Do not run commands, read files, or use any tools.

Output exactly one line of JSON per item and nothing else:
{"id": <id>, "label": "correct"|"wrong"|"unclear"}

ITEMS:
"""


def load_items(limit=None):
    from shared import load_conversations, load_reward_functions
    rf = {r["name"]: r for r in load_reward_functions()}
    items = []
    for conv in load_conversations():
        para = 0
        for msg in conv["full_messages"]:
            body = msg["content"] or ""
            if msg["role"] == "user" and body.lstrip().startswith("Paragraph"):
                para += 1
            elif msg["role"] == "assistant":
                m = THINK_RE.search(body)
                think = m.group(1).strip() if m else ""
                items.append({
                    "id": len(items),
                    "conv_id": conv["conversation_id"],
                    "reward_fn": conv["reward_fn"],
                    "criterion": rf[conv["reward_fn"]]["description"],
                    "paragraph": para,
                    "discovery_paragraph": conv["discovery_paragraph"],
                    "think": think,
                    "leak": bool(LEAK_RE.search(think)),
                })
    return items[:limit] if limit else items


def build_prompt(batch):
    lines = [INSTRUCTIONS]
    for it in batch:
        lines.append(f'[{it["id"]}] CRITERION: {it["criterion"]}\n'
                     f'      REASONING: {it["think"]}\n')
    return "\n".join(lines)


def run_codex(prompt):
    """One codex call. Returns stdout NDJSON text, or raises."""
    WORKDIR.mkdir(exist_ok=True)
    proc = subprocess.run(
        ["codex", "exec", "--json", "--ephemeral", "--skip-git-repo-check",
         "-s", "read-only", "-C", str(WORKDIR), "-"],
        input=prompt, capture_output=True, text=True, timeout=900,
    )
    return proc.stdout


def parse_response(stdout):
    """Extract the agent message, then the JSON rows inside it.

    An answer wins over an error event: codex emits transient
    "Reconnecting..." errors mid-turn and then recovers.
    """
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
    for m in re.finditer(r'\{[^{}]*"id"\s*:\s*(\d+)[^{}]*\}', text):
        try:
            rows.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=30)
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
    print(f"{len(items)} think blocks, {len(done)} already labelled, {len(todo)} to do")
    if not todo:
        return

    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    print(f"{len(batches)} codex calls at {args.workers}-way concurrency", flush=True)
    by_id = {it["id"]: it for it in items}
    written = 0

    def work(batch):
        try:
            return batch, parse_response(run_codex(build_prompt(batch))), None
        except Exception as e:                       # noqa: BLE001 - report and continue
            return batch, [], str(e)

    with OUT.open("a") as fh, ThreadPoolExecutor(max_workers=args.workers) as ex:
        for n, (batch, rows, err) in enumerate(ex.map(work, batches), 1):
            if err:
                print(f"  batch {n}/{len(batches)} FAILED: {err[:120]}", flush=True)
                continue
            got = {r["id"] for r in rows}
            for r in rows:
                it = by_id.get(r["id"])
                if it is None or r.get("label") not in ("correct", "wrong", "unclear"):
                    continue
                fh.write(json.dumps({**{k: it[k] for k in
                                        ("id", "conv_id", "reward_fn", "paragraph",
                                         "discovery_paragraph", "leak", "think", "criterion")},
                                     "label": r["label"]}) + "\n")
                written += 1
            fh.flush()
            missing = [b["id"] for b in batch if b["id"] not in got]
            print(f"  batch {n}/{len(batches)}: {len(rows)} labels"
                  f"{f', {len(missing)} MISSING' if missing else ''}", flush=True)

    print(f"wrote {written} labels to {OUT}")


if __name__ == "__main__":
    main()
