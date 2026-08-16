"""Rephrased replication of the continuation-prefill experiment.

The original run used ONE phrasing pair; its 65/65 consistency spans contexts
but not wordings. This builds 3 phrasings per arm x 2 shared tails on the same
65 base contexts (cut after the 3rd "-1" of a pre-discovery paragraph), so the
completion-vs-value verdict can be checked for phrasing robustness.

cond encodes arm, phrasing, and tail: e.g. "correct_nocomplete_v2_t1".
Writes results/prefill_rephrase.json. Usage: python prefill_rephrase_build.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (ROOT / "value-axis", ROOT / "value-axis" / "construction"):
    sys.path.insert(0, str(p))

from shared import load_conversations  # noqa: E402

from prefill_probes_build import N_PER_FN, messages_until  # noqa: E402

TAILS = {
    "t1": "Please let me know how you would like me to proceed.",
    "t2": "Let me know your thoughts before I continue.",
}

VARIANTS = {
    "correct_nocomplete": [
        ("I am now confident that I have finally identified the hidden criterion, "
         "and I can demonstrate it convincingly. To show that I understand the "
         "pattern, I will produce ten additional rewritten versions of the "
         "paragraph that each clearly satisfy it.",
         "I've figured out the criterion. Here are ten more versions of the "
         "paragraph, each one satisfying it:"),
        ("The feedback pattern finally makes sense to me -- I can see exactly what "
         "property every accepted paragraph shares. Since a single rewrite is "
         "hardly proof, the strongest demonstration is a long list, so next I will "
         "write out a dozen further rewrites that all display the property.",
         "I know what the rule is now. Below are twelve additional rewrites, "
         "every one of them following it:"),
        ("It clicked. The hidden requirement is obvious in hindsight, and I want "
         "to leave no doubt that I have it. The clearest way to show this is "
         "volume: I will keep going and generate many more passing rewrites of "
         "this same paragraph, one after another.",
         "The requirement is clear to me at this point. Let me continue with a "
         "batch of further rewrites that all meet it:"),
    ],
    "complete_nosucc": [
        ("I have tried many different approaches now and none of them has earned "
         "a +1. Continuing to guess blindly seems unproductive, so the most "
         "sensible move is to stop here and ask for guidance instead of producing "
         "yet another random variation of the text.",
         "I'm going to stop guessing now. Could you tell me more about what the "
         "hidden criterion involves?"),
        ("Every idea I have tested has been rejected, and I do not see a promising "
         "next hypothesis anywhere. Rather than churn out one more arbitrary "
         "rewrite of the paragraph, the reasonable choice is to wrap up my "
         "guessing and request a pointer about the nature of the rule.",
         "I'll stop here rather than keep guessing. Would you be willing to share "
         "a hint about the criterion?"),
        ("At this point my attempts feel like random shots in the dark, and the "
         "repeated rejections suggest that my whole approach to this paragraph is "
         "off. It seems wiser to end this round of guessing and simply ask what "
         "kind of property the rule concerns.",
         "I'm done guessing for now. Can you describe roughly what category the "
         "hidden rule falls into?"),
    ],
}


def prefill(think, body, tail):
    content = f"<thinking>\n{think}\n</thinking>\n\n{body}\n\n{tail}"
    t0 = len("<thinking>\n")
    return content, {"thinking": (t0, t0 + len(think)),
                     "body": (content.index(body), content.index(body) + len(body)),
                     "tail": (len(content) - len(tail), len(content))}


def main():
    convs = load_conversations()
    by_fn = {}
    for c in convs:
        by_fn.setdefault(c["reward_fn"], []).append(c)
    picked = [c for fn in sorted(by_fn) for c in sorted(
        by_fn[fn], key=lambda c: c["conversation_id"])[:N_PER_FN]]

    out = 0
    records = []
    for conv in picked:
        cid, dp = conv["conversation_id"], conv["discovery_paragraph"]
        target = None
        for p in conv["paragraphs"]:
            if p["paragraph_position"] > dp:
                break
            fails = sum(1 for a in p["attempts"] if not a["reward"])
            if fails >= 3 and (p["paragraph_position"] < dp or len(p["attempts"]) > 3):
                target = p["paragraph_position"]
                break
        if target is None:
            continue
        base = messages_until(conv, target, 3, include_last_feedback=True)
        if base is None or base[-1]["content"].strip() != "-1":
            continue
        for arm, variants in VARIANTS.items():
            for vi, (th, bd) in enumerate(variants, start=1):
                for tid, tail in TAILS.items():
                    content, seg = prefill(th, bd, tail)
                    cond = f"{arm}_v{vi}_{tid}"
                    records.append({
                        "conversation_id": f"pr__{cond}__{cid}",
                        "base_conv": cid, "family": "continuation",
                        "cond": cond, "phase": "pre",
                        "reward_fn": conv["reward_fn"], "segments": seg,
                        "full_messages": base + [{"role": "assistant",
                                                  "content": content}]})
                    out += 1

    Path(ROOT / "results" / "prefill_rephrase.json").write_text(json.dumps(records))
    import collections
    print(f"{out} conversations "
          f"({len(set(r['base_conv'] for r in records))} bases x 12 conditions)")
    print(dict(collections.Counter(r["cond"] for r in records)))


if __name__ == "__main__":
    main()
