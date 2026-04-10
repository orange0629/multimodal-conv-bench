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
    "final_answer": "<the correct final answer to the last question>",
    "reasoning_chain": "<step-by-step explanation of the correct reasoning across turns>",
    "key_difficulty": "<what makes this hard for a model — what single-turn reasoning would miss>"
  }
}

Rules:
- Always start with a user turn that includes an image.
- Alternate user/assistant roles strictly.
- image_description must be detailed enough to generate or retrieve a real image (include objects, spatial layout, colors, lighting, style).
- The final user turn must contain a concrete answerable question.
- Do NOT include the answer in the conversation — only in ground_truth.
- Output ONLY the JSON object, no markdown fences or extra text.
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
    "user_template": """Scenario domain: {scenario}
Number of turns (user+assistant pairs): {n_turns}
Mode: {mode}

Generate a multi-turn conversation in this domain.
{seed_image_hint}
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
    "user_template": """Scenario domain: {scenario}
Number of turns (user+assistant pairs): {n_turns}
Mode: {mode}

Generate a multi-turn conversation in this domain.
{seed_image_hint}
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
    "user_template": """Scenario domain: {scenario}
Number of turns (user+assistant pairs): {n_turns}
Mode: {mode}

Generate a multi-turn conversation in this domain.
{seed_image_hint}
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
    "user_template": """Scenario domain: {scenario}
Number of turns (user+assistant pairs): {n_turns}
Mode: {mode}

Generate a multi-turn conversation in this domain.
{seed_image_hint}
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
    "user_template": """Scenario domain: {scenario}
Number of turns (user+assistant pairs): {n_turns}
Mode: {mode}

Generate a multi-turn conversation in this domain.
{seed_image_hint}
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
    "user_template": """Scenario domain: {scenario}
Number of turns (user+assistant pairs): {n_turns}
Mode: {mode}

Generate a multi-turn conversation in this domain.
{seed_image_hint}
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


def build_messages(taxonomy: str, scenario: str, n_turns: int, mode: str,
                   seed_image_description: str | None = None) -> list[dict]:
    """Build the [system, user] messages for a synthesis request."""
    taxonomy = TAXONOMY_ALIASES.get(taxonomy, taxonomy)
    if taxonomy not in TAXONOMIES:
        raise ValueError(f"Unknown taxonomy '{taxonomy}'. Choose from: {list(TAXONOMIES)}")

    tmpl = TAXONOMIES[taxonomy]

    if seed_image_description:
        seed_hint = f"Seed image description (use this as the starting visual context):\n  {seed_image_description}\n"
    else:
        seed_hint = ""

    mode_note = (
        "text-only (describe each image in the image_description field; no real images exist)"
        if mode == "text-only"
        else "with-images (image_description should match a real provided seed image)"
    )

    user_content = tmpl["user_template"].format(
        scenario=scenario,
        n_turns=n_turns,
        mode=mode_note,
        seed_image_hint=seed_hint,
        schema=OUTPUT_SCHEMA,
    )

    return [
        {"role": "system", "content": tmpl["system"]},
        {"role": "user", "content": user_content},
    ]
