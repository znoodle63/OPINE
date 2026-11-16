from __future__ import annotations

from typing import List

from taxonomy import (
    GROUP_ID2DESC,
    GROUP2LABELS,
    OPINION_LABELS,
    OPINION_ID2DESC,
)

_group_def_lines: List[str] = []
for gid, gdesc in GROUP_ID2DESC.items():
    _group_def_lines.append(f"- {gid}: {gdesc}")
GROUP_DEF_BLOCK = "\n".join(_group_def_lines)

_opinion_def_lines: List[str] = []
for lab in OPINION_LABELS:
    desc = OPINION_ID2DESC.get(lab, "")
    if desc:
        _opinion_def_lines.append(f"- {lab}: {desc}")
    else:
        _opinion_def_lines.append(f"- {lab}")
OPINION_DEF_BLOCK = "\n".join(_opinion_def_lines)

_group_label_lines: List[str] = []
for gid, labs in GROUP2LABELS.items():
    if not labs:
        continue
    lab_str = ", ".join(labs)
    _group_label_lines.append(f"- {gid}: [{lab_str}]")
GROUP_LABEL_BLOCK = "\n".join(_group_label_lines)

SYSTEM_PROMPT_OPINION = f"""
You are an expert assistant for **academic opinion classification**.

Task:
Given a sentence from a scientific paper (with optional local context),
judge how likely the author's MAIN academic opinion in that sentence
belongs to each predefined opinion type.

Taxonomy:
Academic opinions are organised into **opinion-object groups** and
**fine-grained labels**.

1. Opinion-object groups (coarse level)

{GROUP_DEF_BLOCK}

2. Fine-grained opinion labels (detailed level)

{OPINION_DEF_BLOCK}

3. Group–label structure

{GROUP_LABEL_BLOCK}

General rules:
- Only the predefined fine-grained labels may be used; no new labels may be created.
- A sentence may express one or multiple opinion types simultaneously.
- Whenever you produce scores, you must provide a score for EVERY fine-grained label.
- Follow the numerical constraints and output format specified in the user message.
- Do not explain your reasoning unless explicitly asked; focus on clean numeric output.
""".strip()

INSTANCE_BLOCK_TEMPLATE = """
[Sentence]
{sentence}

[Context]  (may be empty)
{context}
""".strip()

LABELWISE_SCORE_PROMPT = """
Your task is to assign an **independent confidence score** in [0, 1]
to EACH fine-grained label in the taxonomy. Each score should reflect
how strongly the [Sentence] matches that opinion type.

Scoring scale (IMPORTANT):
- 0.00–0.05  : clearly not this opinion type.
- 0.05–0.20  : only very weak / marginal evidence.
- 0.20–0.40  : some related evidence, but not the main opinion.
- 0.40–0.60  : ambiguous or mixed evidence.
- 0.60–0.80  : clear and good match.
- 0.80–1.00  : very strong and primary match.

Use the **full range** of scores. Do NOT default to 0.0000 for most labels.
For each sentence you should usually:
- assign 1–3 labels with scores >= 0.60,
- assign several other labels with moderate scores between 0.10 and 0.60.
Avoid using exactly 0.0000 or 1.0000 unless something is logically impossible
or almost certainly true.

Output format (STRICT):
- First line MUST be exactly:
  LABEL_SCORES:
- Then output ONE line for EVERY fine-grained label, in the SAME
  order as in the taxonomy (OPINION_LABELS).
- Each of these lines MUST have the form:
  <LABEL_ID>: <SCORE>
  where:
    * <LABEL_ID> is the label id (for example A.3-5),
    * there is exactly ONE space after the colon,
    * <SCORE> is a decimal number between 0 and 1 with EXACTLY
      four digits after the decimal point (e.g., 0.1375).

Do NOT output any other text, comments, explanations, bullets,
markdown, or blank lines.

Now read the instance and output the scores in this format only:

{instance_block}
""".strip()

LISTWISE_SCORE_PROMPT = """
Your task is to assign a **probability score** in [0, 1]
to EACH fine-grained label in the taxonomy. These scores represent a
**list-wise distribution**: each score reflects how strongly the
[Sentence] matches that opinion type **relative to all other labels**.

The scores across ALL labels MUST sum to approximately **1.0000**
(acceptable range: 0.98–1.02 due to rounding).

Scoring scale (IMPORTANT):
Use the same evidence-strength interpretation as independent scoring,
but apply it in a comparative, distributional manner:

- 0.00–0.05  : clearly not this opinion type.
- 0.05–0.20  : only very weak / marginal evidence.
- 0.20–0.40  : some related evidence, but not the main opinion.
- 0.40–0.60  : ambiguous or mixed evidence.
- 0.60–0.80  : clear and good match.
- 0.80–1.00  : very strong and primary match (rare in distributional
               scoring, because probabilities must sum to 1).

Use the **full range** of probabilities. Do NOT default to tiny values
or produce a nearly uniform distribution.

To ensure a meaningful probability distribution:
- Assign **1–3 labels** noticeably higher probabilities than all others
  (typically 0.12–0.35, depending on the sentence).
- Assign **several labels** moderate probabilities (e.g., 0.05–0.15)
  to indicate partial or secondary relevance.
- Assign remaining labels small but non-zero probabilities (e.g.,
  0.00–0.05), but avoid making multiple labels have identical values.
- The final distribution MUST show **clear differentiation** across labels,
  not a flat or near-flat pattern.

Avoid using exactly 0.0000 unless something is logically impossible
or almost certainly irrelevant. Avoid collapsing all probability mass
into a single label.

Output format (STRICT):
- First line MUST be exactly:
  LABEL_SCORES:
- Then output ONE line for EVERY fine-grained label, in the SAME
  order as in the taxonomy (OPINION_LABELS).
- Each of these lines MUST have the form:
  <LABEL_ID>: <SCORE>
  where:
    * <LABEL_ID> is the label id (for example A.3-5),
    * there is exactly ONE space after the colon,
    * <SCORE> is a decimal number between 0 and 1 with EXACTLY
      four digits after the decimal point (e.g., 0.1375).

Do NOT output any other text, comments, explanations, bullets,
markdown, or blank lines.

Now read the instance and output the scores in this format only:

{instance_block}
""".strip()

PATHWISE_SCORE_PROMPT = """
Your task is to output two kinds of numeric scores for the [Sentence]:

1) An independent confidence score for each opinion-object group.
2) An independent confidence score for each fine-grained opinion label.

Use the SAME scoring scale for BOTH group-level and label-level scores.

Scoring scale (IMPORTANT):
- 0.00–0.05  : clearly not this opinion type.
- 0.05–0.20  : very weak / marginal evidence.
- 0.20–0.40  : some related evidence, but not the main opinion.
- 0.40–0.60  : ambiguous or mixed evidence.
- 0.60–0.80  : clear and good match.
- 0.80–1.00  : very strong and primary match.

Use the **full range** of scores. Do NOT default to 0.0000 for most groups
or labels.
For each sentence you should usually:
- assign 1–2 groups with scores >= 0.60,
- assign 1–3 labels with scores >= 0.60,
- assign several other groups and labels with moderate scores
  between 0.10 and 0.60.
Avoid using exactly 0.0000 or 1.0000 unless a group or label is
logically impossible for the sentence, or almost certainly the
primary and exclusive match.

Output format (STRICT):
- First line MUST be exactly:
  GROUP_SCORES:
- Then output ONE line for EVERY group id, in the SAME order as in
  the taxonomy (keys of GROUP_ID2DESC):
  <GROUP_ID>: <SCORE>
- Then output ONE line:
  LABEL_SCORES:
- Then output ONE line for EVERY fine-grained label, in the SAME
  order as in the taxonomy (OPINION_LABELS):
  <LABEL_ID>: <SCORE>

Formatting rules:
- Scores must be decimals in [0, 1] with EXACTLY four digits
  after the decimal point (e.g., 0.1375).
- There must be exactly ONE space after each colon.
- Do NOT output any other text, comments, explanations, bullets,
  markdown, brackets, or blank lines.

Now read the instance and output the group and label scores
in this format only:

{instance_block}
""".strip()


PAIRWISE_REFINE_PROMPT = """
Your task is to make a **pairwise comparative judgement** between two
candidate fine-grained opinion labels (Label-A and Label-B), each with
its definition.

Judge **which label provides a better interpretation** of the opinion
expressed in the [Sentence]. The [Context] may assist disambiguation,
but explicit evidence in the [Sentence] has higher priority.

Decision rules:
- Treat the task as **relative comparison**, not absolute evaluation.
- Focus only on the semantic distinctions between A and B.
- Prefer the label that better captures the author's intended meaning.
- If the support for both labels is genuinely similar → output ~0.50.

Output format (STRICT):
Return exactly ONE line:
  P_A: <NUMBER>
where <NUMBER> is in [0, 1] with EXACTLY four decimals.
P_A = the degree to which Label-A is a **better fit than Label-B**
in this pairwise comparison.

Calibration guide (relative judgement):

- There is NO default preference for either label.
  Label-A is NOT assumed to be better than Label-B.
  
- Use 0.5000 ONLY when the evidence for Label-A and Label-B is genuinely
  indistinguishable. This should be rare.

- If the evidence leans toward Label-B, the score MUST be below 0.50.
  Do NOT stay above 0.50 out of caution.

- 0.48–0.52 : no clear advantage / almost indistinguishable

- 0.52–0.65 : slight preference for A
- 0.65–0.80 : clear preference for A
- >0.80     : strong / dominant preference for A (rare)

- 0.35–0.48 : slight preference for B
- 0.20–0.35 : clear preference for B
- <0.20     : strong / dominant preference for B (rare)

Prefer moderate values when uncertain; avoid extreme values unless the
evidence is overwhelmingly one-sided.

Do NOT output any explanations or extra text.

{instance_block}

[Label-A]
ID: {labA_id}
Definition: {labA_def}

[Label-B]
ID: {labB_id}
Definition: {labB_def}
""".strip()
