"""
Prompt templates for both pipeline stages:
  Stage 1 — Batch blueprint generation (one LLM call → N diverse blueprints)
  Stage 2 — Dialogue instantiation (one blueprint → one conversation)
"""
from __future__ import annotations

from dialogvis.models import TaxonomyType


# ===========================================================================
# STAGE 1: Batch blueprint generation
# One prompt per taxonomy. One LLM call returns a JSON array of N blueprints.
# ===========================================================================

_BLUEPRINT_OUTPUT_FORMAT = """\
For each blueprint output exactly this JSON structure:
{
  "id": <int, 1-indexed within this batch>,
  "taxonomy": "<taxonomy name>",
  "scenario": "<specific domain + situation, e.g. 'Dermatology: classifying a forearm lesion'>",
  "num_turns": <int between 3 and 6>,
  "visual_sequence": [
    {
      "turn": <int, 1-indexed user turn this image appears in>,
      "image_desc": "<detailed, standalone description of what the image shows — write as a T2I prompt: specify modality, subject, lighting, composition, key visual details. NEVER reference other images.>",
      "role": "<see taxonomy-specific roles below>",
      "reasoning_effect": "<what a model should think or conclude after seeing this image>"
    }
  ],
  "dialogue_arc": "<one sentence: how the model's belief/understanding should evolve turn by turn>",
  "ground_truth": "<the single correct final answer>",
  "distractor": "<the wrong answer a model would commit to after turn 1 alone>",
  "final_question": "<the question posed to the evaluated model at the end of the conversation>",
  "single_turn_solvable": false,
  "why_sequential": "<one sentence: why turn order matters — what the sequential delivery enables that simultaneous delivery cannot>"
}

Output a JSON array of {count} blueprints. No other text, no markdown fences."""


_DIVERSITY_HEADER = """\
You are designing a multi-turn visual dialogue benchmark dataset.

Your task: Generate {count} DIVERSE blueprints for the taxonomy below.
Each blueprint must be a scenario where sequential image delivery is \
CONSTITUTIVE — collapsing all images into a single turn must make \
the task impossible or trivially easier.

TAXONOMY: {taxonomy_name}
DEFINITION: {definition}

DIVERSITY REQUIREMENTS — across your {count} blueprints, vary ALL of:
- Scenario domain (do NOT repeat the same domain twice)
- Number of turns (between 3 and 6, do not use the same value more than twice)
- {diversity_axes}

Scenario domains to draw from (use freely, invent others):
medical imaging, satellite imagery, food authenticity, art authentication,
wildlife identification, geology/mineralogy, forensic evidence,
sports replay analysis, cooking process, fashion counterfeit detection,
architecture damage assessment, plant disease diagnosis, underwater imagery,
crime scene reconstruction, material science, dermatology, astronomy,
security surveillance, archaeology, industrial inspection, agriculture,
marine biology, meteorology, pathology, numismatics, gemology...

"""


# ---------------------------------------------------------------------------
# Per-taxonomy batch prompts
# ---------------------------------------------------------------------------


_TAXONOMY_SPECS: dict[TaxonomyType, dict] = {

    TaxonomyType.BELIEF_REVISION: dict(
        taxonomy_name="Belief Revision under Visual Evidence",
        definition=(
            "Early images are ambiguous or misleading. Subsequent images "
            "provide clarifying evidence. The model must update earlier "
            "interpretations rather than anchoring on initial judgments."
        ),
        diversity_axes=(
            "Type of misleading evidence (lighting artifact, partial view, "
            "visual similarity to another class, scale confusion, out-of-context "
            "framing, color cast, motion blur, unusual angle, etc.)\n"
            "- Strength of initial false belief (slightly wrong vs completely wrong)\n"
            "- Number of revision steps (single correction vs gradual revision)"
        ),
        role_guide=(
            "Roles to use: \"misleading\" (turn 1 image that triggers the wrong belief), "
            "\"clarifying\" (image that begins to overturn it), "
            "\"confirming\" (optional final image that seals the correct answer), "
            "\"contradicting\" (image that directly refutes the prior belief).\n"
            "Rule: the distractor must be a natural conclusion from turn 1 alone. "
            "The ground_truth must require seeing the clarifying image."
        ),
    ),

    TaxonomyType.INCREMENTAL_STATE: dict(
        taxonomy_name="Incremental State Tracking",
        definition=(
            "Each turn introduces a slightly modified version of a scene. "
            "The model must detect, accumulate, and reason about changes over time, "
            "maintaining a persistent mental model that no single image fully captures."
        ),
        diversity_axes=(
            "Type of tracked attribute (count, position, color, condition, quantity, "
            "temperature, fill level, assembly progress, damage extent, etc.)\n"
            "- Number of state updates (2–4 intermediate changes)\n"
            "- Presence of distractor changes (irrelevant changes that should be ignored)"
        ),
        role_guide=(
            "Roles to use: \"establishing\" (turn 1 baseline state), "
            "\"update\" (each subsequent image showing a state change), "
            "\"distractor_update\" (a change irrelevant to the final question).\n"
            "Rule: ground_truth must be the FINAL state, not the initial. "
            "Distractor must be the initial state or a plausible intermediate."
        ),
    ),

    TaxonomyType.ENTITY_TRACKING: dict(
        taxonomy_name="Cross-Turn Entity Tracking and Reference Resolution",
        definition=(
            "The model must identify and re-identify the same entities across "
            "different images and dialogue turns, especially when the user refers "
            "to prior visual content through anaphoric expressions."
        ),
        diversity_axes=(
            "Type of entity (person, vehicle, animal, object, landmark, specimen)\n"
            "- Type of anaphoric reference used in the final question "
            "(\"the one who...\", \"that same vehicle\", \"the specimen from earlier\")\n"
            "- What makes re-identification challenging "
            "(similar-looking distractors, changed context, partial occlusion, time gap)"
        ),
        role_guide=(
            "Roles to use: \"establishing\" (introduces 2+ entities), "
            "\"reappearance\" (entity seen again in new context), "
            "\"partial_view\" (entity partially visible, requires prior memory), "
            "\"distractor_entity\" (a new similar-looking entity introduced as a foil).\n"
            "Rule: the final_question must use an anaphoric reference. "
            "The distractor must be a plausible wrong entity identification."
        ),
    ),

    TaxonomyType.TEMPORAL_CAUSAL: dict(
        taxonomy_name="Temporal and Causal Reasoning from Sequential Images",
        definition=(
            "Images depict stages of a process or event, potentially arriving "
            "out of chronological order. The model must reconstruct the temporal "
            "sequence and reason about causal relationships."
        ),
        diversity_axes=(
            "Type of process (biological, mechanical, chemical, geological, social, "
            "culinary, forensic, astronomical, developmental)\n"
            "- Degree of shuffle (fully reversed, one swap, random permutation)\n"
            "- Whether the question asks for ordering, causation, or prediction"
        ),
        role_guide=(
            "Roles to use: \"out-of-order\" for all images (label their actual "
            "chronological position in reasoning_effect, e.g. 'This is actually stage 3 of 4').\n"
            "Rule: explicitly state in the scenario that images are NOT in order. "
            "The final_question asks for the correct sequence OR causal explanation. "
            "The distractor is a plausible but incorrect ordering or causal attribution."
        ),
    ),

    TaxonomyType.INTERACTIVE_DIALOGUE: dict(
        taxonomy_name="Interactive Visual Dialogue",
        definition=(
            "The image provided at each turn is a direct response to the model's "
            "prior output — the user annotates, crops, or re-photographs based on "
            "the model's suggestion. This tests coherent reasoning when the visual "
            "input stream is shaped by the dialogue itself."
        ),
        diversity_axes=(
            "Type of interaction shaping the next image (zoom-in on pointed area, "
            "annotated copy, different angle requested, measurement added, filter applied)\n"
            "- Whether the model's preliminary conclusion is wrong (requires correction) "
            "or incomplete (requires refinement)\n"
            "- Domain (technical inspection, medical consultation, art critique, "
            "navigation, cooking, forensics)"
        ),
        role_guide=(
            "Roles to use: \"initial\" (turn 1 image), "
            "\"response_to_model\" (image causally triggered by the model's prior response), "
            "\"correction\" (image that overturns a wrong preliminary conclusion).\n"
            "Rule: each image after turn 1 must be justified by what the model said. "
            "Include at least one \"correction\" image. "
            "The distractor is the wrong preliminary conclusion."
        ),
    ),

    TaxonomyType.STRATEGIC_ACQUISITION: dict(
        taxonomy_name="Strategic Information Acquisition",
        definition=(
            "The model must identify gaps in its visual understanding and "
            "proactively request specific additional images. Tests not just "
            "reasoning over given images, but recognizing what is missing."
        ),
        diversity_axes=(
            "Type of missing information (different viewpoint, higher resolution, "
            "different modality, wider context, interior view, measurement reference)\n"
            "- Whether there is a suboptimal but plausible alternative request\n"
            "- Domain (medical triage, device repair, species identification, "
            "structural assessment, authentication, navigation)"
        ),
        role_guide=(
            "Roles to use: \"insufficient\" (turn 1 — genuinely ambiguous, cannot be resolved alone), "
            "\"optimal_response\" (the most informative follow-up image), "
            "\"suboptimal_response\" (a less useful image the model might have requested instead).\n"
            "Rule: include exactly one suboptimal_response image. "
            "The ground_truth includes both the correct answer AND the optimal acquisition strategy. "
            "The distractor is the conclusion reached via the suboptimal path. "
            "Set evaluation_type to dynamic in your mind — the model's questions matter."
        ),
    ),
}


def build_batch_blueprint_prompt(taxonomy: TaxonomyType, count: int) -> str:
    """Return the full prompt for generating `count` diverse blueprints for a taxonomy."""
    spec = _TAXONOMY_SPECS[taxonomy]
    header = _DIVERSITY_HEADER.format(
        count=count,
        taxonomy_name=spec["taxonomy_name"],
        definition=spec["definition"],
        diversity_axes=spec["diversity_axes"],
    )
    # Substitute {count} in the output format without triggering KeyError on JSON braces
    output_fmt = _BLUEPRINT_OUTPUT_FORMAT.replace("{count}", str(count))
    return header + spec["role_guide"] + "\n\n" + output_fmt


# ===========================================================================
# STAGE 2: Dialogue instantiation prompt (unchanged)
# ===========================================================================

DIALOGUE_SCHEMA = """\
{
  "id": "<uuid-v4>",
  "blueprint_id": "<id from the input blueprint>",
  "taxonomy": "<same as blueprint>",
  "scenario": "<same as blueprint>",
  "turns": [
    {
      "turn_id": 1,
      "speaker": "user",
      "text": "<natural user message text>",
      "image": {
        "id": "img_001",
        "description": "<copied verbatim from blueprint visual_sequence[0].image_desc>",
        "is_seed": true
      }
    },
    {
      "turn_id": 2,
      "speaker": "assistant",
      "text": "<assistant response — must reflect the reasoning_effect of the prior image>"
    },
    {
      "turn_id": 3,
      "speaker": "user",
      "text": "<user message introducing next image>",
      "image": {
        "id": "img_002",
        "description": "<copied verbatim from blueprint visual_sequence[1].image_desc>",
        "is_seed": false
      }
    }
  ],
  "final_question": "<copied verbatim from blueprint>",
  "mcq_options": [
    {"label": "A", "text": "<ground_truth text>", "is_correct": true},
    {"label": "B", "text": "<distractor text>", "is_correct": false},
    {"label": "C", "text": "<plausible wrong option>", "is_correct": false},
    {"label": "D", "text": "<plausible wrong option>", "is_correct": false}
  ],
  "ground_truth": "<copied from blueprint>",
  "reasoning_chain": "<step-by-step explanation of why the correct answer follows from the visual sequence>",
  "single_turn_solvable": false,
  "why_sequential": "<copied from blueprint>",
  "difficulty": "<easy | medium | hard — judge based on how subtle the visual evidence is>",
  "evaluation_type": "static"
}"""

DIALOGUE_SYSTEM_PROMPT = """\
You are a dialogue writer for a vision-language model benchmark. \
You will receive a structured blueprint and must instantiate it into a \
realistic multi-turn conversation.

Write natural dialogue that follows the blueprint's reasoning arc exactly. \
The conversation should feel like a genuine expert consultation while guiding \
the model's reasoning through the visual sequence as planned.

Output schema:
{schema}

Guidelines:
- User messages: natural and concise, as typed by a real person sharing images.
- Assistant messages: reflect the reasoning_effect of the image just shown — \
commit to interpretations, ask follow-up questions, or revise as specified.
- Do NOT let the assistant reach the correct answer before the final turn.
- The last user message must pose the final_question from the blueprint.
- Copy image descriptions VERBATIM from the blueprint — do not paraphrase.
- Shuffle the correct MCQ label (not always "A"). All options must be \
plausible to someone who saw only the first image.
- reasoning_chain must trace which image at which turn provided the decisive evidence.

CRITICAL: Output ONLY the raw JSON object. No markdown fences, no commentary.
""".format(schema=DIALOGUE_SCHEMA)
