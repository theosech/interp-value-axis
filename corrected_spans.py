"""Corrected token labeler for the value-axis construction.

Upstream extract_activations.py locates each attempt's modified paragraph with
``formatted.find(modified[:150], search_from)``. When the model's first edit falls
after character 150 (61% of dp+1 attempts), that needle equals the ORIGINAL
paragraph's opening, so the span lands on the user's copy of the paragraph — the
wrong turn entirely. Two further bugs follow: the span end ``idx + len(modified)``
overruns into chat markers, and semantic reward-word offsets are computed inside
``modified_text`` but added to a start index pointing at original text.

This module keeps the paper's design (successful attempt of paragraph
discovery_paragraph+1; split on criterion-satisfying tokens; >= MIN_TOKENS_PER_SIDE
tokens per side) and fixes only the localization:

1. The span is the assistant turn's body after the </thinking> block, located
   structurally via turns.attempt_records — never by text search.
2. Semantic reward words are matched inside that same span text, so offsets index
   the text they were computed from.

Return contract matches upstream classify_tokens: [(pos, "before"|"after"|"excluded")].
Import-light on purpose: only `re` and turns.py, so it loads on the Modal image and
locally without the upstream repo on sys.path (the caller passes checkers in).
"""

import re

from turns import attempt_records

MIN_TOKENS_PER_SIDE = 3          # same threshold as upstream extract_activations.py
_THINK_END = ("</think>", "</thinking>")


def target_paragraph_span(formatted, discovery_paragraph):
    """[start, end) of the modified paragraph inside the dp+1 assistant turn.

    Located structurally: the (single, always-rewarded) attempt of the first
    post-discovery paragraph, minus its <thinking> block and leading whitespace.
    Returns None if the attempt is missing or unrewarded (never happens in the
    released corpus, but stay defensive).
    """
    rec = next((r for r in attempt_records(formatted, discovery_paragraph)
                if r["paragraph"] == discovery_paragraph + 1 and r["reward"] is True),
               None)
    if rec is None:
        return None
    body = formatted[rec["assistant_start"]:rec["assistant_end"]]
    cut = max((body.find(t) + len(t) for t in _THINK_END if t in body), default=0)
    while cut < len(body) and body[cut] in " \t\r\n":
        cut += 1
    if cut >= len(body):
        return None
    return rec["assistant_start"] + cut, rec["assistant_end"]


def corrected_classify_tokens(token_strings, offset_mapping, formatted, conv,
                              reward_fn_name, reward_fn_type, reward_labels,
                              conv_idx, syntactic_token_checkers):
    """Label every token "before" / "after" / "excluded" with corrected spans.

    Same semantics as upstream classify_tokens: only tokens inside the successful
    modification of the first post-discovery paragraph are candidates; reward
    positions come from the token-level syntactic checkers or the Claude word
    labels; strictly-before -> "before", strictly-after -> "after".

    `syntactic_token_checkers` is upstream shared.SYNTACTIC_TOKEN_CHECKERS,
    passed in rather than imported so this module has no upstream import.
    """
    n = len(token_strings)
    excluded = [(pos, "excluded") for pos in range(n)]

    span = target_paragraph_span(formatted, conv["discovery_paragraph"])
    if span is None:
        return excluded
    p0, p1 = span

    in_span = [pos for pos in range(n)
               if offset_mapping[pos][0] >= p0 and offset_mapping[pos][1] <= p1
               and offset_mapping[pos][1] > offset_mapping[pos][0]]
    if not in_span:
        return excluded

    if reward_fn_type == "syntactic":
        checker = syntactic_token_checkers[reward_fn_name]
        reward_positions = {pos for pos in in_span if checker(token_strings[pos])}
    else:
        target_para = conv["discovery_paragraph"] + 1
        words = reward_labels.get((reward_fn_name, conv_idx, target_para), [])
        char_spans = []
        for word in words:
            for m in re.finditer(re.escape(word), formatted[p0:p1], re.IGNORECASE):
                char_spans.append((p0 + m.start(), p0 + m.end()))
        reward_positions = {pos for pos in in_span
                            if any(offset_mapping[pos][0] < end and offset_mapping[pos][1] > start
                                   for start, end in char_spans)}
    if not reward_positions:
        return excluded

    before = {p for p in in_span if p < min(reward_positions)}
    after = {p for p in in_span if p > max(reward_positions)}
    if len(before) < MIN_TOKENS_PER_SIDE or len(after) < MIN_TOKENS_PER_SIDE:
        return excluded
    return [(pos, "before" if pos in before else "after" if pos in after else "excluded")
            for pos in range(n)]
