"""Human-adjustable model choices, paths, RGDR parameters, and VLM instructions.

The external preparation function named in DIR prepares base, task, and VLM.
Native model and VLM options use the names supported by the connected environment.
"""

# Native flow model and generation options. Additional settings can include
# resolution, sampler, guidance, precision, and other options supported by the model.
# None resolves from an existing generation record or native defaults.
# Explicit choices must match the original image's recorded configuration.
FLOW_MODEL_CONFIG = {
    "model_id": None,
    "num_steps": None,  # Native updates; the full trajectory has num_steps + 1 points.
    "settings": {},
}

# Initialization seed. Its record belongs to base.sampling_state.
# For an existing base, reuse the recorded initial state and required RNG/sampler state.
seed = None

# Selected VLM and its native loading/request options, including supported
# reasoning controls. The preparation code interprets these options for that VLM.
VLM_CONFIG = {
    "model_id": None,
    "settings": {},
}

# Locations: external preparation code, input files, and output directory.
# integration uses an importable "module:function" reference.
DIR = {
    "integration": None,
    "input": {
        "task_path": None,
        "image_path": None,
        "generation_record_path": None,
    },
    "output": None,
}

# RGDR correction strength and normalization stabilizer.
eta = 0.1
epsilon = 1e-8


def s_t(t: float, T: float) -> float:
    """Temporal weight; t is the current model time and T is its initial time."""
    return 1.0


# Numerical fulfillment scores, judgments, and visual evidence.
DEFAULT_EVALUATION_PROTOCOL = """Evaluate the image against the prompt and checklist.
For each question, identify the intended requirement and assess its fulfillment
using the task description, question semantics, and visible image evidence.
Return a numerical fulfillment score in [0, 1], a judgment of pass, fail, or
uncertain, and a brief description of the supporting visual evidence.
Keep the fulfillment judgment separate from a literal affirmative or negative
answer to the question. Interpret presence and absence checks according to what the prompt requires.
Use the degree of fulfillment for the numerical score. Report uncertain when
visual evidence does not establish a confident pass or fail.
Return per-item evaluations. The program computes the task score."""


# Feedback and paired Target/Reference descriptions.
DEFAULT_REFLECTION_INSTRUCTION = """Reflect on the evaluated image and the task.
Use the checklist evaluation to identify residual errors and describe the
correction needed to satisfy the requested attributes, objects, or relations.
Produce a Target text describing the desired state after the correction and a
Reference text describing the current erroneous state of the same content.
Describe only the erroneous content in Reference and its corresponding
corrected state in Target.
Keep the two texts focused on the same correction target, with the context
needed to locate it. Omit unrelated requirements already satisfied by the image.
Preserve the content that already satisfies the task.
When all requirements are satisfied, return empty strings for both target_text
and reference_text.
Return one structured result containing evaluations, feedback, target_text,
and reference_text. For each checklist item, include its question, judgment,
score, and evidence. Preserve the supplied checklist order."""
