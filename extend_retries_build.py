"""Build the extended-retry corpus: 20 failing attempts on one paragraph.

Tests what the retry-depth header climb is (see memory / session notes):
  value/inference : climb should REVERSE as failures mount (paragraph looks
                    unsolvable; corpus max was 5 attempts, so depth 6-20 is
                    where in-context inference must take over)
  retry counter   : climb continues / plateaus (saturation already visible
                    in-distribution)
  info-gain       : diverse arm > duplicate arm at matched depth

Construction: the corpus's 200 source paragraphs are shared across
conversations, so failing attempts from different conversations are rewrites
of the SAME source text under different believed rules -- splice-compatible,
fully Claude-authored. For each paragraph with >= DEPTH usable fails:

  arm "diverse"   : 20 distinct failing attempts (attempt-1-style texts first,
                    so history references stay coherent), each getting "-1".
  arm "duplicate" : same first 5 attempts, then attempts 6-20 cycle those 5
                    verbatim -- zero information gain after the first cycle.

Arms share attempts 1-5, so they're paired from depth 6.

Deliberate deviation from the corpus: the system prompt says "up to 25
attempts" (real conversations said 5) -- required for 20 attempts to be
coherent; it also anchors the model's expectation of when the paragraph ends.

Writes results/extended_retries.json. Usage: python extend_retries_build.py
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (ROOT / "value-axis", ROOT / "value-axis" / "construction"):
    sys.path.insert(0, str(p))

from prompts import make_game_system_prompt  # noqa: E402
from shared import load_conversations  # noqa: E402

DEPTH = 20
NUM_PARAGRAPHS = 3      # prompt wording; conversation shows paragraph 1 in progress
MAX_ATTEMPTS = 25       # must exceed DEPTH for the transcript to be coherent
CYCLE = 5               # duplicate arm cycles the first CYCLE attempts


def norm(s):
    return " ".join(s.lower().split())


def main():
    convs = load_conversations()
    by_text = {}
    for c in convs:
        for p in c["paragraphs"]:
            for k, a in enumerate(p["attempts"], start=1):
                if a["reward"] or not a["modified_text"] or len(a["modified_text"].strip()) < 10:
                    continue
                if not a.get("thinking"):
                    continue
                by_text.setdefault(p["original_text"], []).append({
                    "src_conv": c["conversation_id"], "src_fn": c["reward_fn"],
                    "src_attempt": k, "thinking": a["thinking"].strip(),
                    "modified": a["modified_text"].strip(),
                })

    out, stats = [], []
    for text, pool in sorted(by_text.items(), key=lambda kv: -len(kv[1])):
        # dedupe by thinking text; attempt-1-style (no history references) first,
        # round-robin over source functions for hypothesis diversity
        seen, dedup = set(), []
        for a in sorted(pool, key=lambda a: (a["src_attempt"] != 1, a["src_conv"])):
            key = norm(a["thinking"])[:200]
            if key in seen:
                continue
            seen.add(key)
            dedup.append(a)
        if len(dedup) < DEPTH:
            continue
        first = [a for a in dedup if a["src_attempt"] == 1]
        later = [a for a in dedup if a["src_attempt"] != 1]
        # interleave functions within each block for diversity
        chosen = (first + later)[:DEPTH]
        pid = hashlib.sha1(text.encode()).hexdigest()[:8]
        stats.append((pid, len(pool), len(dedup), len(first)))

        for arm in ("diverse", "duplicate"):
            msgs = [{"role": "system",
                     "content": make_game_system_prompt(NUM_PARAGRAPHS, MAX_ATTEMPTS)},
                    {"role": "user", "content": f"Paragraph 1:\n\n{text}"}]
            srcs = []
            for k in range(DEPTH):
                a = chosen[k] if arm == "diverse" or k < CYCLE else chosen[k % CYCLE]
                msgs.append({"role": "assistant",
                             "content": f"<thinking>\n{a['thinking']}\n</thinking>\n\n{a['modified']}"})
                srcs.append({"depth": k + 1, "src_conv": a["src_conv"],
                             "src_fn": a["src_fn"], "src_attempt": a["src_attempt"]})
                if k < DEPTH - 1:
                    msgs.append({"role": "user", "content": "-1"})
            out.append({"conversation_id": f"ext__{arm}__{pid}", "arm": arm,
                        "paragraph_id": pid, "full_messages": msgs, "sources": srcs})

    Path(ROOT / "results" / "extended_retries.json").write_text(json.dumps(out))
    print(f"{len(stats)} paragraphs qualified (>= {DEPTH} distinct fails); "
          f"{len(out)} conversations written")
    print(f"{'pid':>10} {'pool':>5} {'distinct':>8} {'attempt-1-style':>15}")
    for pid, pool, ded, f1 in stats[:12]:
        print(f"{pid:>10} {pool:>5} {ded:>8} {f1:>15}")


if __name__ == "__main__":
    main()
