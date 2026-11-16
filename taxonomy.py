"""Discourse-function groups and labels for academic opinion classification."""

# level-1
GROUP_ID2DESC = {
    "A.1": "Opinions about research actors in the scientific community (who contributes what to the field)",
    "A.2": "Opinions about research outputs (the value and impact of concrete scientific products such as papers, models, datasets)",
    "A.3": "Opinions about research objects and problem space (fields, topics, problem formulations, gaps, and challenges)",
    "B.1": "Opinions about research hypotheses formulated in this study",
    "B.2": "Opinions about research methods and methodological design in this study",
    "B.3": "Opinions about experimental schemes and experimental design in this study",
    "B.4": "Opinions about research results produced by this study (their meaning, causes, and comparative performance)",
    "B.5": "Opinions about the overall conclusions of this study, including contributions, limitations, and future research directions"
}

# level-2
OPINION_ID2DESC = {
    "A.1-1": "Opinions on the contribution and influence of research actors (e.g., individual researchers, research groups, institutions), highlighting who has driven the development of a field and in what way",
    "A.2-1": "Opinions on the value and impact of specific research outputs (papers, models, datasets, tools, systems), focusing on their relative advantages, practical usefulness, or pioneering role within a method family",
    "A.3-1": "Opinions on the overall status and development stage of a research field, summarising trends, paradigm shifts, or whether the field is in an emerging, developing, or mature phase",
    "A.3-2": "Opinions on current key research priorities and hotspots, identifying core topics that are repeatedly discussed or highly active in the literature at a given time",
    "A.3-3": "Opinions on mainstream knowledge and consensus regarding a research topic, stating what has been repeatedly verified and is widely accepted in the community about certain methods, assumptions, or phenomena",
    "A.3-4": "Opinions on mainstream technical routes in the field, highlighting which solution strategies or methodological frameworks have become standard or dominant ways of addressing a problem",
    "A.3-5": "Opinions on current research gaps and insufficiencies, pointing out under-explored areas, missing aspects, or limitations of existing work that leave important problems unresolved",
    "A.3-6": "Opinions on current research difficulties and challenges, revealing inherent obstacles or technical bottlenecks that make a task particularly hard to solve, even when it is already widely studied",
    "A.3-7": "Opinions on the selection and formulation of research problems, explaining how general needs or gaps are concretised into specific core questions, technical targets, or mechanisms to be investigated, and how the problem is delimited",
    "A.3-8": "Opinions on the value of a research field, topic, or specific problem, judging why it is worth studying in terms of theoretical importance or practical relevance",
    "B.1-1": "Opinions on research hypotheses, understood as reasoned but unverified expectations or predictions about variable relations, mechanisms, or effects, typically expressed with verbs such as hypothesise, assume, or posit",
    "B.2-1": "Opinions on the rationality of research method design, explaining why a particular methodological strategy, architecture, or component is appropriate for the research goals, and how its structure matches the task needs",
    "B.3-1": "Opinions on the rationality of experimental scheme design, justifying why experiments are organised in a specific way, including choices of datasets, baselines, control settings, and evaluation protocols, and how these choices support credible comparison and validation",
    "B.4-1": "Opinions that interpret research results or provide causal explanations, describing what the observed findings mean and why they occur, often using cautious epistemic verbs such as indicate, suggest, or imply, or causal markers such as due to and attributable to",
    "B.4-2": "Opinions that compare research results and judge superiority or effectiveness, emphasising how a method performs better, worse, or comparably to baselines or prior work, typically using comparative expressions like outperform or better than",
    "B.5-1": "Opinions on the contribution and value of this study as a whole, clarifying what knowledge gaps it fills, which technical bottlenecks it breaks, or what new perspectives it introduces, often marked by terms like novel, pioneering, or valuable",
    "B.5-2": "Opinions on the limitations and insufficiencies of this study, explicitly stating constraints related to data, model capacity, experimental settings, task assumptions, or resources, and thus delineating the scope in which the conclusions remain valid",
    "B.5-3": "Opinions on future improvements and development directions, proposing concrete extensions of the current work (e.g., larger data, new models, broader scenarios) and/or higher-level future directions for the field, often introduced by expressions such as future work or should be further explored",
}

GROUP2LABELS = {
    "A.1": ["A.1-1"],
    "A.2": ["A.2-1"],
    "A.3": ["A.3-1", "A.3-2", "A.3-3", "A.3-4", "A.3-5", "A.3-6", "A.3-7", "A.3-8"],
    "B.1": ["B.1-1"],
    "B.2": ["B.2-1"],
    "B.3": ["B.3-1"],
    "B.4": ["B.4-1", "B.4-2"],
    "B.5": ["B.5-1", "B.5-2", "B.5-3"],
}


OPINION_LABELS = sorted(OPINION_ID2DESC.keys())
NUM_LABELS = len(OPINION_LABELS)

# int id <-> fine-grained label string
OPINION_ID2LABEL = {i: lab for i, lab in enumerate(OPINION_LABELS)}
LABEL2OPINION_ID = {lab: i for i, lab in OPINION_ID2LABEL.items()}

# fine-grained label -> group (A.1, A.2, ..., B.5)
LABEL2GROUP = {}
for g, labs in GROUP2LABELS.items():
    for lab in labs:
        LABEL2GROUP[lab] = g

GROUP_IDS = list(GROUP_ID2DESC.keys())
GROUP_ID2IDX = {g: i for i, g in enumerate(GROUP_IDS)}
IDX2GROUP_ID = {i: g for g, i in GROUP_ID2IDX.items()}
NUM_GROUPS = len(GROUP_IDS)


# Confusable label pairs specified in Section 5.1.
CONFUSABLE_PAIRS = {
    frozenset(["A.3-1", "A.3-2"]),
    frozenset(["A.3-1", "A.3-3"]),
    frozenset(["A.3-1", "A.3-4"]),
    frozenset(["A.3-3", "A.3-4"]),
    frozenset(["A.3-5", "A.3-6"]),
    frozenset(["A.3-5", "A.3-7"]),
    frozenset(["A.3-6", "A.3-7"]),
    frozenset(["B.4-1", "B.4-2"]),
    frozenset(["B.2-1", "B.3-1"]),
    frozenset(["B.5-1", "B.5-2"]),
    frozenset(["B.5-2", "B.5-3"]),
}
