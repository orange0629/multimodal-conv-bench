from __future__ import annotations

import uuid
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class TaxonomyType(str, Enum):
    INCREMENTAL_STATE     = "incremental_state_tracking"
    BELIEF_REVISION       = "belief_revision"
    ENTITY_TRACKING       = "cross_turn_entity_tracking"
    TEMPORAL_CAUSAL       = "temporal_causal_reasoning"
    INTERACTIVE_DIALOGUE  = "interactive_visual_dialogue"
    STRATEGIC_ACQUISITION = "strategic_information_acquisition"


# ---------------------------------------------------------------------------
# Blueprint models
# ---------------------------------------------------------------------------

class ImageSpec(BaseModel):
    """Specifies one image in the visual sequence of a blueprint."""
    turn: int                  # which user turn this image belongs to (1-indexed)
    image_desc: str            # standalone, vivid description for a T2I model
    role: str                  # e.g. misleading | clarifying | confirming | establishing |
                               #       tracking | requesting | revealing | out-of-order
    reasoning_effect: str      # what the model should think / do after seeing this image


class Blueprint(BaseModel):
    """
    Stage-1 output: a structured reasoning plan for one benchmark item.
    Does NOT contain dialogue text — only the design of the visual sequence,
    the reasoning arc, and the evaluation question.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    taxonomy: TaxonomyType
    scenario: str                        # e.g. "Dermatology: Classifying a skin lesion"
    num_turns: int                       # total turns (user + assistant combined)
    visual_sequence: list[ImageSpec]     # one entry per user-turn image
    dialogue_arc: str                    # high-level description of how reasoning evolves
    ground_truth: str                    # correct final answer
    distractor: str                      # primary wrong answer (appears in MCQ)
    additional_distractors: list[str] = Field(default_factory=list)  # for 4-option MCQ
    final_question: str                  # question posed to the evaluated model at the end
    single_turn_solvable: bool           # False = collapsing turns destroys the task
    why_sequential: str                  # justification for why sequential delivery matters
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Conversation (Stage-2 output)
# ---------------------------------------------------------------------------

class MCQOption(BaseModel):
    label: str      # "A", "B", "C", "D"
    text: str
    is_correct: bool


class ImageRef(BaseModel):
    """A reference to an image in the conversation."""
    id: str
    description: str     # copied from blueprint ImageSpec.image_desc
    is_seed: bool = False
    source_path: str | None = None


class Turn(BaseModel):
    turn_id: int
    speaker: Literal["user", "assistant"]
    text: str
    image: ImageRef | None = None


class Conversation(BaseModel):
    """
    Stage-2 output: the full multi-turn dialogue instantiated from a Blueprint.
    Includes MCQ for evaluation.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    blueprint_id: str                    # links back to the Blueprint
    taxonomy: TaxonomyType
    scenario: str
    turns: list[Turn]

    # Evaluation
    final_question: str
    mcq_options: list[MCQOption]         # typically 3–4 options; exactly one is_correct
    ground_truth: str                    # correct answer text (matches the is_correct option)
    reasoning_chain: str                 # step-by-step explanation for the correct answer
    single_turn_solvable: bool
    why_sequential: str

    difficulty: Literal["easy", "medium", "hard"]
    evaluation_type: Literal["static", "dynamic"]
    num_images: int = 0
    metadata: dict = Field(default_factory=dict)

    def model_post_init(self, __context: object) -> None:
        self.num_images = sum(1 for t in self.turns if t.image is not None)


# ---------------------------------------------------------------------------
# Job spec
# ---------------------------------------------------------------------------

class GenerationJob(BaseModel):
    """
    One unit of work: generate `count` diverse blueprints for a taxonomy.
    The LLM invents all scenarios and varies num_turns automatically.
    """
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    taxonomy: TaxonomyType
    count: int = Field(default=10, ge=1, le=50)  # blueprints to generate per call
    hints: str | None = None                      # optional extra instructions
