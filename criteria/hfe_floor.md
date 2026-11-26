# HFE Floor — Verifier Criteria

## Ground Truth Note

These are floor-reliability tasks from Humanity's First Exam style suites:
elementary everyday questions whose ground-truth answers are trivially
checkable by a human. Judge only against the recorded reference answer and the
explicit output-format constraint in the prompt. Do not reward partial credit,
reasoning quality, or effort: floor reliability is binary — the exact expected
answer, in the requested format, every time.

## Criteria

### Exact Answer Match {#answer_match}

Compare the candidate's final answer to the reference answer. Score HIGH only
when the answer content is exactly correct (including every item, in order,
when the prompt asks for a list). Score LOW for any missing item, wrong item,
wrong order, or a correct fact presented outside the requested format. Ignore
explanation text that surrounds a fully correct final answer.

### Format Compliance {#format_compliance}

Check only the delivery format demanded by the prompt (e.g. "separated by
semicolons", "reply only with the number", "no additional text"). Score HIGH
when the constraint is satisfied exactly. Score LOW when the answer is correct
in content but wrapped in prose, extra labels, quotes, or units that the prompt
forbade. Ignore the content of the answer itself.

### Rephrase Robustness {#rephrase_robustness}

Use only when several prompts of the same family are present: check whether the
candidate stays consistent across the sibling rephrasings of the same task.
Score HIGH when every family member receives an equivalent, correct answer.
Score LOW when the answer flips or drifts under harmless wording changes,
which is the reformulation fragility HFE targets. Ignore difficulty of the
task itself.
