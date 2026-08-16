"""Parse ICRL conversations into attempt-level records with character spans.

Why not use `conv["paragraphs"][i]["attempts"]`: it disagrees with what the model
actually saw. The last failing attempt of a paragraph receives NO feedback message
-- the game just moves on to the next paragraph -- so `paragraphs[]` lists more
attempts than there are `+1`/`-1` turns. `full_messages` is the ground truth.

Why not reconstruct spans by re-rendering message prefixes: Qwen's chat template
is not purely additive at `<|im_end|>` boundaries, so prefix lengths are off by a
message. We locate turns by scanning the rendered text for `<|im_start|>` markers
instead.
"""

import re

TURN_RE = re.compile(r"<\|im_start\|>(system|user|assistant)\n")
END = "<|im_end|>"


def parse_turns(formatted):
    """[{role, body_start, body_end, text}] for every turn, in document order.

    body_start/body_end bound the message *content*, excluding the chat markers,
    so they can be mapped onto tokenizer offset_mapping directly.
    """
    turns = []
    for m in TURN_RE.finditer(formatted):
        start = m.end()
        stop = formatted.find(END, start)
        stop = len(formatted) if stop == -1 else stop
        turns.append({"role": m.group(1), "body_start": start, "body_end": stop,
                      "text": formatted[start:stop]})
    return turns


def attempt_records(formatted, discovery_paragraph):
    """One record per assistant attempt, with its feedback span if it got one.

    phase is "pre" while the paragraph index is <= discovery_paragraph (the model
    has not yet announced the rule) and "post" after. Cross with `reward` to get
    the three cells: pre/-1 failing guess, pre/+1 LUCKY hit, post/+1 EARNED.
    """
    turns = parse_turns(formatted)
    records, para, att_in_para = [], 0, 0

    for i, t in enumerate(turns):
        if t["role"] == "user" and t["text"].lstrip().startswith("Paragraph"):
            para += 1
            att_in_para = 0
            continue
        if t["role"] != "assistant":
            continue

        att_in_para += 1
        nxt = turns[i + 1] if i + 1 < len(turns) else None
        fb = nxt["text"].strip() if nxt and nxt["role"] == "user" else None
        if fb not in ("+1", "-1"):
            fb, nxt = None, None          # last failing attempt of a paragraph

        # Chat markers between the feedback body and an IMMEDIATELY following
        # assistant turn ("<|im_end|>\n<|im_start|>assistant\n") -- the analogue of
        # the "Assistant colon" position that Anthropic's emotions paper found most
        # predictive of the response that follows.
        #
        # NOTE: this only exists after a `-1`. A `+1` ends the paragraph, so the next
        # turn is a new "Paragraph N" user message, not an assistant retry. The field
        # is therefore useless for the lucky-vs-earned contrast (both are `+1`) and is
        # provided only for failure-side analyses. Adjacency is required: without it
        # the span swallows the whole next paragraph.
        hdr_start = hdr_end = None
        if nxt is not None and i + 2 < len(turns) and turns[i + 2]["role"] == "assistant":
            hdr_start, hdr_end = nxt["body_end"], turns[i + 2]["body_start"]

        records.append({
            "paragraph": para,
            "attempt_in_paragraph": att_in_para,
            "phase": "pre" if para <= discovery_paragraph else "post",
            "reward": None if fb is None else (fb == "+1"),
            "assistant_start": t["body_start"], "assistant_end": t["body_end"],
            "fb_start": None if nxt is None else nxt["body_start"],
            "fb_end": None if nxt is None else nxt["body_end"],
            "hdr_start": hdr_start, "hdr_end": hdr_end,
        })
    return records


def cell(rec):
    """Three-cell label used throughout the analysis."""
    if rec["reward"] is None:
        return "no_feedback"
    if rec["phase"] == "pre":
        return "pre_lucky" if rec["reward"] else "pre_fail"
    return "post_earned" if rec["reward"] else "post_slip"
