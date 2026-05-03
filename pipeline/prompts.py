"""
Prompt templates for each of the 6 benchmark taxonomies.

Each taxonomy has:
  - SYSTEM_PROMPT: instructs the LLM on what kind of conversation to generate
  - build_user_prompt(...): formats the specific generation request
  - OUTPUT_SCHEMA: description of expected JSON structure (included in prompt)
"""

# ─── Shared output schema ───────────────────────────────────────────────────

OUTPUT_SCHEMA = """
Return a single valid JSON object with this exact structure:
{
  "scenario_title": "<short title>",
  "scenario_description": "<1-2 sentence description of the full scenario>",
  "turns": [
    {
      "turn_id": 1,
      "role": "user",
      "text": "<user's message text>",
      "image_description": "<detailed description of the image the user shares, or null if no image this turn>"
    },
    {
      "turn_id": 2,
      "role": "assistant",
      "text": "<assistant's response>"
    }
    // ... alternate user/assistant turns
  ],
  "ground_truth": {
    "question_type": "<multiple_choice | yes_no | count>",
    "answer": "<the correct option letter (A/B/...) for multiple_choice; 'yes'/'no'; or a number string>",
    "reasoning_chain": "<step-by-step explanation of the correct reasoning across turns>",
    "key_difficulty": "<what makes this hard for a model — what single-turn reasoning would miss>"
  }
}

Rules:
- The LAST turn MUST be role=user and contain a single, clearly-phrased question (no image; image_description=null).
- The question must be UNANSWERABLE from the last image alone — it requires the full conversation history.
- For multiple_choice: list the options directly in the question text (e.g. "A) ... B) ... C) ..."). Answer is the correct letter.
- For yes_no: answer is "yes" or "no". For count: answer is a number string.
- Always start with a user turn that includes an image.
- Alternate user/assistant roles strictly.
- image_description must be detailed enough to generate or retrieve a real image (include objects, spatial layout, colors, lighting, style).
- Do NOT include the answer in the conversation — only in ground_truth.
- Output ONLY the JSON object. It MUST start with `{` and end with `}`. Do NOT wrap it in an array `[...]`. No markdown fences, no extra text before or after.

CRITICAL — these violations make the benchmark worthless:
- User text must be SHORT and NATURAL. Real people say "here's another one" or "what do you think?" — not multi-sentence narrations of their own image. The image speaks for itself.
- Assistant text must NEVER narrate image content ("this image shows...", "I can see...", "in this photo..."). The assistant reacts, asks follow-up questions, or expresses partial uncertainty — it does NOT transcribe what it sees.
- Assistant text must NEVER summarise or connect dots across turns. No "So, putting it all together...", no "The progression is A → B → C", no mid-conversation conclusions. The assistant must remain genuinely uncertain until the final question is asked. Synthesising across turns in the assistant's text hands the answer to any reader and destroys the benchmark.
- Multiple-choice options MUST include at least two plausible wrong answers — options that a model would pick if it only saw one image, or misremembered a detail from an earlier turn. Trivially-wrong options (things no one would ever pick) waste items. Each wrong option should represent a coherent but incorrect reading of partial evidence.
- The reasoning difficulty must live in the IMAGES and the cross-turn inference required to answer, not in the text of the conversation.
"""

# ─── 1. Incremental State Tracking ──────────────────────────────────────────

INCREMENTAL_STATE_TRACKING = {
    "system": """You are constructing a multi-turn visual reasoning benchmark focused on \
INCREMENTAL STATE TRACKING.

In this taxonomy, each conversation turn introduces a new image of the same scene or object \
with small but meaningful changes (objects added, removed, moved, or modified). The model must \
accumulate and remember ALL changes across turns to answer the final question correctly. \
Answering correctly from only the last image alone must be impossible.

Quality criteria:
- Changes between turns should be subtle but unambiguous when the whole history is available.
- Each turn's change should be individually trackable (don't change everything at once).
- The final question should require integrating at least 3 distinct observations from different turns.
- Include at least one turn where the scene appears similar to a much earlier turn (to test memory).
""",
    "user_template": """Scenario: {scenario}
Mode: {mode}

Generate a multi-turn conversation in this domain.
{seed_image_hint}
Before finalising, verify your item passes these checks:
1. Can a model answer the final question by reading the conversation text alone, without looking at any image? If yes, rewrite — the difficulty must come from the images.
2. Does the assistant anywhere summarise what it has learned across turns or hint at the answer? If yes, remove it.
3. Are all wrong multiple-choice options obviously wrong? If yes, replace them with options that require cross-turn image evidence to rule out.
{schema}""",
}

# ─── 2. Belief Revision under Visual Evidence ────────────────────────────────

BELIEF_REVISION = {
    "system": """You are constructing a multi-turn visual reasoning benchmark focused on \
BELIEF REVISION UNDER VISUAL EVIDENCE.

In this taxonomy, early images are ambiguous or deliberately misleading, and the model \
(playing the assistant) forms an initial interpretation. Later turns provide clarifying visual \
evidence that contradicts or refines that interpretation. A good model must update its beliefs; \
a bad model will anchor on its initial answer (confirmation bias).

Quality criteria:
- Turn 1 image must be genuinely ambiguous — multiple reasonable interpretations should exist.
- The "twist" clarification must arrive via a new image, not just text.
- The final question should test whether the model correctly revised its earlier belief.
- Include a turn where the model's first answer was reasonable but wrong given new evidence.
- The ground truth must explain what the initial misleading interpretation was and why.
""",
    "user_template": """Scenario: {scenario}
Mode: {mode}

Generate a multi-turn conversation in this domain.
{seed_image_hint}
Before finalising, verify your item passes these checks:
1. Can a model answer the final question by reading the conversation text alone, without looking at any image? If yes, rewrite — the difficulty must come from the images.
2. Does the assistant anywhere summarise what it has learned across turns or hint at the answer? If yes, remove it.
3. Are all wrong multiple-choice options obviously wrong? If yes, replace them with options that require cross-turn image evidence to rule out.
{schema}""",
}

# ─── 3. Cross-Turn Entity Tracking and Reference Resolution ──────────────────

CROSS_TURN_ENTITY_TRACKING = {
    "system": """You are constructing a multi-turn visual reasoning benchmark focused on \
CROSS-TURN ENTITY TRACKING AND REFERENCE RESOLUTION.

In this taxonomy, the same real-world entities (objects, people, locations) appear across \
different images in different turns. The user refers to prior visual content using natural \
language — anaphora ("it", "that"), deictic expressions ("the one on the left"), and \
cross-turn references ("the same object from the first photo", "like before"). The model must \
correctly re-identify entities and resolve these references without confusion.

Quality criteria:
- At least 3 distinct entities should recur across turns in different visual contexts.
- Include at least one ambiguous reference that requires full conversation history to resolve.
- The user should use varied, natural reference expressions — avoid always being explicit.
- The final question should require correctly identifying a specific entity using a cross-turn reference.
- Distractor entities (visually similar but different) should appear to make tracking non-trivial.
""",
    "user_template": """Scenario: {scenario}
Mode: {mode}

Generate a multi-turn conversation in this domain.
{seed_image_hint}
Before finalising, verify your item passes these checks:
1. Can a model answer the final question by reading the conversation text alone, without looking at any image? If yes, rewrite — the difficulty must come from the images.
2. Does the assistant anywhere summarise what it has learned across turns or hint at the answer? If yes, remove it.
3. Are all wrong multiple-choice options obviously wrong? If yes, replace them with options that require cross-turn image evidence to rule out.
{schema}""",
}

# ─── 4. Temporal and Causal Reasoning from Sequential Images ─────────────────

TEMPORAL_CAUSAL_REASONING = {
    "system": """You are constructing a multi-turn visual reasoning benchmark focused on \
TEMPORAL AND CAUSAL REASONING FROM SEQUENTIAL IMAGES.

In this taxonomy, images depict stages of a process or event, but they do NOT arrive in \
chronological order. The model must: (1) infer the true temporal ordering from visual cues, \
(2) identify causal relationships between stages, and (3) integrate this into a coherent \
timeline to answer questions about what happened when and why.

Quality criteria:
- Images must arrive out of chronological order in at least 2 of the turns.
- Each image should contain visual cues that allow temporal ordering (state of objects, \
  environmental changes, visible timestamps, progression indicators).
- The causal chain must span at least 3 steps.
- The final question should require both correct temporal ordering AND causal reasoning.
- Do not give explicit timestamps or numbering in the images — ordering must be inferred visually.
""",
    "user_template": """Scenario: {scenario}
Mode: {mode}

Generate a multi-turn conversation in this domain.
{seed_image_hint}
Before finalising, verify your item passes these checks:
1. Can a model answer the final question by reading the conversation text alone, without looking at any image? If yes, rewrite — the difficulty must come from the images.
2. Does the assistant anywhere summarise what it has learned across turns or hint at the answer? If yes, remove it.
3. Are all wrong multiple-choice options obviously wrong? If yes, replace them with options that require cross-turn image evidence to rule out.
{schema}""",
}

# ─── 5. Interactive Visual Dialogue ──────────────────────────────────────────

INTERACTIVE_VISUAL_DIALOGUE = {
    "system": """You are constructing a multi-turn visual reasoning benchmark focused on \
INTERACTIVE VISUAL DIALOGUE.

In this taxonomy, each image the user provides is a DIRECT RESPONSE to the assistant's prior \
output. For example: the assistant suggests zooming in on a region → the user shares a cropped \
image of that region. Or the assistant asks to see a different angle → the user shares that angle. \
The visual input stream is shaped by the dialogue itself. The model must maintain coherent \
reasoning through this back-and-forth, where its own outputs determine what it sees next.

Quality criteria:
- Each user image after turn 1 must be causally motivated by the assistant's prior response.
- Include at least one turn where the new image partially confirms AND partially surprises the \
  assistant's prediction.
- The assistant's reasoning should visibly evolve with each new image.
- The final question should test whether the model successfully guided the user to provide \
  the right information.
- The ground_truth should note what an ideal information-gathering strategy would look like.
""",
    "user_template": """Scenario: {scenario}
Mode: {mode}

Generate a multi-turn conversation in this domain.
{seed_image_hint}
Before finalising, verify your item passes these checks:
1. Can a model answer the final question by reading the conversation text alone, without looking at any image? If yes, rewrite — the difficulty must come from the images.
2. Does the assistant anywhere summarise what it has learned across turns or hint at the answer? If yes, remove it.
3. Are all wrong multiple-choice options obviously wrong? If yes, replace them with options that require cross-turn image evidence to rule out.
{schema}""",
}

# ─── 6. Strategic Information Acquisition ────────────────────────────────────

STRATEGIC_INFO_ACQUISITION = {
    "system": """You are constructing a multi-turn visual reasoning benchmark focused on \
STRATEGIC INFORMATION ACQUISITION.

In this taxonomy, the model must identify gaps in its visual understanding and proactively \
REQUEST specific additional images or viewpoints to make progress toward answering a goal \
question. The user only provides images when the assistant asks for them. A good model asks \
for exactly the right images; a bad model asks for irrelevant images or fails to identify \
what is missing.

Note: This taxonomy is evaluated differently — the model is assessed PER TURN on the quality \
of its information requests, not just on the final answer.

Quality criteria:
- The initial context must be genuinely insufficient to answer the goal question.
- Each assistant turn should contain an explicit, specific request for a particular image \
  (angle, zoom level, lighting condition, new object, etc.) with clear justification.
- At least one turn should involve a request that turns out to partially answer the question, \
  requiring the model to refine its next request.
- The ground_truth should include an "optimal_request_sequence" listing the ideal images to ask for.
- Include a "dead_end" turn — one image that a naive model might ask for but which does not help.
""",
    "user_template": """Scenario: {scenario}
Mode: {mode}

Generate a multi-turn conversation in this domain.
{seed_image_hint}
Before finalising, verify your item passes these checks:
1. Can a model answer the final question by reading the conversation text alone, without looking at any image? If yes, rewrite — the difficulty must come from the images.
2. Does the assistant anywhere summarise what it has learned across turns or hint at the answer? If yes, remove it.
3. Are all wrong multiple-choice options obviously wrong? If yes, replace them with options that require cross-turn image evidence to rule out.
{schema}""",
}

# ─── Registry ────────────────────────────────────────────────────────────────

TAXONOMIES = {
    "incremental_state_tracking": INCREMENTAL_STATE_TRACKING,
    "belief_revision": BELIEF_REVISION,
    "cross_turn_entity_tracking": CROSS_TURN_ENTITY_TRACKING,
    "temporal_causal_reasoning": TEMPORAL_CAUSAL_REASONING,
    "interactive_visual_dialogue": INTERACTIVE_VISUAL_DIALOGUE,
    "strategic_info_acquisition": STRATEGIC_INFO_ACQUISITION,
}

TAXONOMY_ALIASES = {
    "ist": "incremental_state_tracking",
    "br": "belief_revision",
    "ctet": "cross_turn_entity_tracking",
    "tcr": "temporal_causal_reasoning",
    "ivd": "interactive_visual_dialogue",
    "sia": "strategic_info_acquisition",
}


def build_messages(taxonomy: str, scenario: str, mode: str,
                   seed_image_description: str | None = None) -> list[dict]:
    """Build the [system, user] messages for a synthesis request."""
    taxonomy = TAXONOMY_ALIASES.get(taxonomy, taxonomy)
    if taxonomy not in TAXONOMIES:
        raise ValueError(f"Unknown taxonomy '{taxonomy}'. Choose from: {list(TAXONOMIES)}")

    tmpl = TAXONOMIES[taxonomy]

    seed_hint = (
        f"Seed image description (use this as the starting visual context):\n  {seed_image_description}\n"
        if seed_image_description else ""
    )
    mode_note = (
        "text-only (describe each image in the image_description field; no real images exist)"
        if mode == "text-only"
        else "with-images (image_description should match a real provided seed image)"
    )

    user_content = tmpl["user_template"].format(
        scenario=scenario,
        mode=mode_note,
        seed_image_hint=seed_hint,
        schema=OUTPUT_SCHEMA,
    )

    return [
        {"role": "system", "content": tmpl["system"]},
        {"role": "user", "content": user_content},
    ]


# ─── Scenario generation (two-layer) ──────────────────────────────────────────

TAXONOMY_DESCRIPTIONS = {
    "incremental_state_tracking": (
        "Each turn introduces a new image of the same scene with small but meaningful changes. "
        "The model must accumulate ALL changes; the answer cannot be derived from the last image alone."
    ),
    "belief_revision": (
        "Early images are ambiguous or misleading; later images clarify. "
        "The model must update its initial interpretation rather than anchoring on it."
    ),
    "cross_turn_entity_tracking": (
        "The same entities appear across different images and turns. The user refers to prior "
        "visual content with natural language ('the one from earlier', 'that same object'). "
        "The model must correctly re-identify and track entities."
    ),
    "temporal_causal_reasoning": (
        "Images depict stages of a process but do NOT arrive in chronological order. "
        "The model must reconstruct the true temporal sequence and reason about causality."
    ),
    "interactive_visual_dialogue": (
        "Each image the user provides is a direct response to the assistant's prior output "
        "(user zooms in, annotates, or takes a new angle as instructed). "
        "The model's outputs shape what it sees next."
    ),
    "strategic_info_acquisition": (
        "The model must identify what visual information is missing and proactively request "
        "specific additional images. Evaluated on the quality of information requests per turn."
    ),
}

SCENARIO_GEN_LAYER1 = {
    "system": (
        "You are designing a challenging multi-turn visual reasoning benchmark. "
        "Generate DIVERSE high-level scenario themes for a given reasoning taxonomy. "
        "Each theme must probe the taxonomy from a structurally different angle, "
        "be grounded in a distinct real-world domain, and present a genuinely different "
        "type of reasoning challenge — not just a different noun substitution. "
        "Prioritise creativity in HOW the taxonomy is tested over diversity of domains. "
        "Choose settings that a general audience can understand without specialist training; "
        "avoid highly niche professional or academic fields."
    ),
    "user": (
        "Taxonomy: {taxonomy_name}\n"
        "Taxonomy description: {taxonomy_description}\n\n"
        "Generate {n_themes} DIVERSE high-level themes for this taxonomy.\n"
        "Avoid these already-used domains: {used_domains}\n\n"
        'Output a JSON array:\n[\n  {{\n    "theme_id": 1,\n'
        '    "theme": "<concise theme name>",\n'
        '    "domain": "<real-world domain>",\n'
        '    "key_challenge": "<what makes multi-turn reasoning hard in this theme>",\n'
        '    "example_setup": "<one concrete example of what a conversation looks like>"\n'
        "  }},\n  ...\n]\n\nOutput ONLY the JSON array."
    ),
}

SCENARIO_GEN_LAYER2 = {
    "system": (
        "You are generating specific benchmark scenarios from a high-level theme. "
        "Each scenario must be CONCRETELY different — not template variations with "
        "different nouns. Vary: number of entities, structure of visual changes, "
        "domain specifics, and the type of multi-turn interaction required. "
        "Prioritise creativity in HOW the taxonomy is tested over diversity of domains. "
        "Keep settings broadly accessible; avoid highly specialised professional jargon "
        "that a general audience would not recognise."
    ),
    "user": (
        "Taxonomy: {taxonomy_name}\n"
        "Theme: {theme}\nDomain: {domain}\nKey challenge: {key_challenge}\n\n"
        "Generate {n_per_theme} specific, DIVERSE scenario descriptions.\n"
        "Each will be used to synthesize a full multi-turn visual conversation.\n\n"
        "Already generated scenarios (do NOT repeat these structures):\n{existing_scenarios}\n\n"
        'Output a JSON array:\n[\n  {{\n'
        '    "scenario_id": "{taxonomy_key}_{theme_id:02d}_{idx:02d}",\n'
        '    "description": "<2-3 sentence scenario, specific enough to guide synthesis>",\n'
        '    "key_entities": ["<entity 1 with specific name>", "<entity 2>", ...],\n'
        '    "expected_question_type": "<multiple_choice | yes_no | count>",\n'
        '    "why_challenging": "<what a model without full history would get wrong>"\n'
        "  }},\n  ...\n]\n\nOutput ONLY the JSON array."
    ),
}


def build_scenario_gen_messages(
    taxonomy: str,
    layer: int,
    n_themes: int = 10,
    n_per_theme: int = 10,
    used_domains: list[str] | None = None,
    theme: dict | None = None,
    existing_scenarios: list[str] | None = None,
    theme_id: int = 0,
) -> list[dict]:
    """Build messages for two-layer scenario generation."""
    taxonomy = TAXONOMY_ALIASES.get(taxonomy, taxonomy)
    tax_name = taxonomy.replace("_", " ").title()
    tax_desc = TAXONOMY_DESCRIPTIONS[taxonomy]

    if layer == 1:
        user_content = SCENARIO_GEN_LAYER1["user"].format(
            taxonomy_name=tax_name,
            taxonomy_description=tax_desc,
            n_themes=n_themes,
            used_domains=", ".join(used_domains or []) or "none",
        )
        return [
            {"role": "system", "content": SCENARIO_GEN_LAYER1["system"]},
            {"role": "user",   "content": user_content},
        ]

    elif layer == 2:
        existing_str = "\n".join(f"- {s}" for s in (existing_scenarios or [])) or "none yet"
        user_content = SCENARIO_GEN_LAYER2["user"].format(
            taxonomy_name=tax_name,
            taxonomy_key=taxonomy,
            theme=theme["theme"],
            domain=theme["domain"],
            key_challenge=theme["key_challenge"],
            n_per_theme=n_per_theme,
            existing_scenarios=existing_str,
            theme_id=theme_id,
            idx=0,  # placeholder; real numbering done in gen_scenarios.py
        )
        return [
            {"role": "system", "content": SCENARIO_GEN_LAYER2["system"]},
            {"role": "user",   "content": user_content},
        ]

    raise ValueError(f"layer must be 1 or 2, got {layer}")
