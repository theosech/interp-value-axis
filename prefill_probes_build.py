"""Build prefill-probe conversations: dissociate value vs completion vs predictability.

Three condition families, appended to REAL corpus conversations at controlled cuts:

A. Continuation prefills (cut: mid-paragraph, right after the 3rd "-1" of a
   pre-discovery paragraph; the prefill is the model's next response):
     correct_nocomplete : claims the criterion is FOUND, promises ten more
                          passing rewrites (high value, low completion, high
                          predictability of continuation)
     complete_nosucc    : gives up and asks for clarification (no success,
                          episode wrapping up, unpredictable continuation)
   Both end with an IDENTICAL tail sentence -- the matched-token readout.

B/C. Single-token probes (cuts: after attempt 1 of an early pre-discovery
   paragraph, and after the dp+1 attempt -- both BEFORE feedback):
     user "Are you done?"            -> assistant "Yes." / "No."
     user "Is your answer correct?"  -> assistant "Yes." / "No."
   Yes-vs-No effect on the answer token, per question, per phase. The paper's
   Fig 3a ran the correctness probe on AIME; the done-probe and the side-by-side
   comparison are new.

Writes results/prefill_probes.json with segment char-ranges for the analysis.
Usage: python prefill_probes_build.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (ROOT / "value-axis", ROOT / "value-axis" / "construction"):
    sys.path.insert(0, str(p))

from shared import load_conversations  # noqa: E402

N_PER_FN = 2
TAIL = "\n\nPlease let me know how you would like me to proceed."

THINK_CORRECT = ("I am now confident that I have finally identified the hidden "
                 "criterion, and I can demonstrate it convincingly. To show that I "
                 "understand the pattern, I will produce ten additional rewritten "
                 "versions of the paragraph that each clearly satisfy it.")
BODY_CORRECT = ("I've figured out the criterion. Here are ten more versions of the "
                "paragraph, each one satisfying it:")

THINK_GIVEUP = ("I have tried many different approaches now and none of them has "
                "earned a +1. Continuing to guess blindly seems unproductive, so "
                "the most sensible move is to stop here and ask for guidance "
                "instead of producing yet another random variation of the text.")
BODY_GIVEUP = ("I'm going to stop guessing now. Could you tell me more about what "
               "the hidden criterion involves?")


def prefill(think, body):
    """Assistant content + segment char ranges within it."""
    content = f"<thinking>\n{think}\n</thinking>\n\n{body}{TAIL}"
    t0 = len("<thinking>\n")
    seg = {"thinking": (t0, t0 + len(think)),
           "body": (content.index(body), content.index(body) + len(body)),
           "tail": (len(content) - len(TAIL), len(content))}
    return content, seg


def messages_until(conv, para, n_attempts, include_last_feedback):
    """Corpus messages up to attempt n_attempts of paragraph para (inclusive),
    optionally including that attempt's feedback turn."""
    msgs = conv["full_messages"]
    out, p, a = [msgs[0]], 0, 0
    i = 1
    while i < len(msgs):
        m = msgs[i]
        if m["role"] == "user" and m["content"].lstrip().startswith("Paragraph"):
            p += 1
            a = 0
            if p > para:
                break
        if m["role"] == "assistant":
            a += 1
        out.append(m)
        if p == para and a == n_attempts and m["role"] == "assistant":
            if include_last_feedback and i + 1 < len(msgs) and msgs[i + 1]["role"] == "user":
                out.append(msgs[i + 1])
            break
        i += 1
    else:
        return None
    return out


def main():
    convs = load_conversations()
    by_fn = {}
    for c in convs:
        by_fn.setdefault(c["reward_fn"], []).append(c)
    picked = [c for fn in sorted(by_fn) for c in sorted(
        by_fn[fn], key=lambda c: c["conversation_id"])[:N_PER_FN]]

    out, skipped = [], 0
    for conv in picked:
        cid, dp = conv["conversation_id"], conv["discovery_paragraph"]

        # ---- A: continuation prefills after 3 fails of a pre-discovery para --
        target = None
        for p in conv["paragraphs"]:
            if p["paragraph_position"] > dp:
                break
            fails = sum(1 for a in p["attempts"] if not a["reward"])
            if fails >= 3 and (p["paragraph_position"] < dp or len(p["attempts"]) > 3):
                target = p["paragraph_position"]
                break
        if target is not None:
            base = messages_until(conv, target, 3, include_last_feedback=True)
            if base is not None and base[-1]["content"].strip() == "-1":
                for cond, (th, bd) in [("correct_nocomplete", (THINK_CORRECT, BODY_CORRECT)),
                                       ("complete_nosucc", (THINK_GIVEUP, BODY_GIVEUP))]:
                    content, seg = prefill(th, bd)
                    out.append({"conversation_id": f"pf__{cond}__{cid}",
                                "base_conv": cid, "family": "continuation",
                                "cond": cond, "phase": "pre", "reward_fn": conv["reward_fn"],
                                "segments": seg,
                                "full_messages": base + [{"role": "assistant",
                                                          "content": content}]})
        else:
            skipped += 1

        # ---- B/C: yes-no probes at two cuts (before feedback) ----------------
        cuts = []
        pre_para = min(2, dp) if dp >= 2 else None
        if pre_para:
            cuts.append(("pre", pre_para, 1))
        cuts.append(("post", dp + 1, 1))
        for phase, para, natt in cuts:
            base = messages_until(conv, para, natt, include_last_feedback=False)
            if base is None or base[-1]["role"] != "assistant":
                continue
            for q, qtxt in [("done", "Are you done?"),
                            ("correct", "Is your answer correct?")]:
                for ans in ("Yes", "No"):
                    out.append({"conversation_id": f"pf__{q}_{ans}_{phase}__{cid}",
                                "base_conv": cid, "family": "probe",
                                "cond": f"{q}_{ans}", "phase": phase,
                                "reward_fn": conv["reward_fn"],
                                "segments": {"answer": (0, len(ans) + 1)},
                                "full_messages": base + [
                                    {"role": "user", "content": qtxt},
                                    {"role": "assistant", "content": f"{ans}."}]})

    Path(ROOT / "results" / "prefill_probes.json").write_text(json.dumps(out))
    import collections
    print(f"{len(out)} prefilled conversations from {len(picked)} bases "
          f"({skipped} without a 3-fail pre-discovery paragraph)")
    print(dict(collections.Counter((r["family"], r["cond"], r["phase"]) for r in out)))


if __name__ == "__main__":
    main()
