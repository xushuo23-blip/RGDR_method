"""Data formats, flow operations, VLM adapter, and preparation helpers.

Model choices, inputs, VLM instructions, and correction parameters are read from
config.py. The RGDR algorithm and complete rectification pass are in RGDR.py.
"""

from copy import deepcopy
from dataclasses import dataclass
import importlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import config

State = Any       # One latent or other native model state at one time.
Image = Any       # One completed image in the integration's image format.
Condition = Any   # Native conditioning, which may contain several components.
TimeSchedule = Callable[[float, float], float]  # s_t(t, T).

# 1. Task requirements.


@dataclass(frozen=True)
class TaskSpec:
    """What to generate and how the VLM should evaluate it."""

    prompt: str
    checklist: Tuple[str, ...]
    evaluation_protocol: Optional[str] = None


# 2. Trajectory records.


@dataclass(frozen=True)
class TrajectoryPoint:
    """One time paired with one state snapshot from a trajectory."""

    time: float
    state: State


# 3. The original generation record.


@dataclass(frozen=True)
class GenerationRecord:
    """Original image and trajectory with their bound flow environment.

    flow owns the recorded model identity, native settings, original prompt,
    and native sampling operations. sampling_state stores the seed and native
    RNG/sampler state required to restart from trajectory[0].state.
    """

    flow: "FlowModel"
    image: Image
    trajectory: Tuple[TrajectoryPoint, ...]  # Initial state and all accepted boundaries.
    sampling_state: Optional[Mapping[str, Any]] = None


# 4. The VLM request and its evaluation/reflection result.


@dataclass(frozen=True)
class ReflectionRequest:
    """Visible inputs to the VLM evaluation and reflection."""

    image: Image
    prompt: str
    checklist: Tuple[str, ...]
    evaluation_protocol: str
    reflection_instruction: str


@dataclass(frozen=True)
class ItemEvaluation:
    """Requirement-fulfillment judgment, numerical score, and visual evidence."""

    question: str
    judgment: str
    score: float
    evidence: str


@dataclass(frozen=True)
class ReflectionResult:
    """Checklist evaluation, feedback, and paired correction texts.

    Empty target_text and reference_text indicate no correction target.
    task_score is computed locally from the per-item numerical scores.
    """

    evaluations: Tuple[ItemEvaluation, ...]
    feedback: str
    target_text: str
    reference_text: str

    @property
    def task_score(self) -> float:
        return 100.0 * sum(item.score for item in self.evaluations) / len(self.evaluations)


# 5. The RGDR settings and final output.


@dataclass(frozen=True)
class RGDRSettings:
    """Additional correction parameters used by one RGDR pass."""

    eta: float
    epsilon: float
    s_t: TimeSchedule


@dataclass(frozen=True)
class RGDRResult:
    """New image and trajectory, original base, and correction settings."""

    image: Image
    trajectory: Tuple[TrajectoryPoint, ...]
    base: GenerationRecord
    settings: RGDRSettings

    @property
    def endpoint(self) -> State:
        return self.trajectory[-1].state


# 6. Operations of the bound flow environment and the selected VLM.

VelocityField = Callable[[float, State], State]


@dataclass(frozen=True)
class FlowModel:
    """Bind original generation records and connected native operations.

    Create flow = FlowModel(...) with the original model identity, settings,
    prompt text, and the connected functions below. Each function is already
    bound to the same native model environment and generation context; the
    wrapper keeps those operations together for RGDR to call.

    model_settings is copied independently of the runtime configuration and
    excludes the seed, which belongs to GenerationRecord.sampling_state.
    All fields use the same state representation, native time, sign, shape,
    and units. Freeze model parameters and disable inference gradient recording
    in the integration's tensor framework.
    """

    # Recorded context of the original image.
    model_id: str
    model_settings: Mapping[str, Any]
    prompt: str  # Original prompt text; native encoding stays inside the connected operations.

    # Independently read the currently loaded model, effective options, and grid.
    # generation_settings includes num_steps and the initialization seed when used.
    # Return ordinary serializable option values, using the recorded option names.
    generation_model_id: Callable[[], str]
    generation_settings: Callable[[], Mapping[str, Any]]
    sampling_times: Callable[[], Sequence[float]]

    # Compute the complete native velocity with the original context and guidance.
    base_velocity: Callable[[State, float], State]
    # Encode the correction texts and evaluate them at the same state and time.
    encode_condition: Callable[[str], Condition]
    conditional_velocity: Callable[[State, float, Condition], State]
    # One sample's Euclidean norm and an independent state snapshot.
    norm: Callable[[State], float]
    copy_state: Callable[[State], State]

    # Prepare a fresh native sampling pass and restore its recorded start state.
    # RNG state is saved after latent initialization and before sampling preparation
    # or the first update; restore it before replaying preparation that uses RNG.
    restore_sampling_state: Callable[[Optional[Mapping[str, Any]]], None]
    # Call the original pipeline/solver with the supplied velocity at its native
    # point of use. Preserve its updates, internal stages, stochastic operations,
    # and checked options; return initial and accepted-boundary state snapshots.
    integrate: Callable[[VelocityField, State, Sequence[float]], Tuple[TrajectoryPoint, ...]]
    # Decode the endpoint with the same native context.
    decode: Callable[[State], Image]

    def __post_init__(self) -> None:
        # Keep the original options separate from mutable runtime options.
        object.__setattr__(self, "model_settings", deepcopy(dict(self.model_settings)))


class VLMAdapter:
    """Connected VLM call, request preparation, and response validation.

    call_model performs the selected model or service call and parses its answer
    into ReflectionResult. Supply a connected function or an existing object's
    bound reflection method.
    """

    def __init__(
        self, call_model: Callable[[ReflectionRequest], ReflectionResult]
    ) -> None:
        if not callable(call_model):
            raise TypeError("call_model must be a connected VLM callable.")
        self._call_model = call_model

    def reflect(self, request: ReflectionRequest) -> ReflectionResult:
        """Call the connected VLM and validate its structured response."""
        reflection = self._call_model(request)
        self._validate_response(reflection, request.checklist)
        return reflection

    def reflect_image(
        self,
        base: GenerationRecord,
        task: TaskSpec,
        *,
        reflection_instruction: Optional[str] = None,
    ) -> ReflectionResult:
        """Evaluate the original image once and return its structured feedback."""
        if reflection_instruction is None:
            reflection_instruction = config.DEFAULT_REFLECTION_INSTRUCTION
        if not isinstance(reflection_instruction, str) or not reflection_instruction.strip():
            raise ValueError("reflection_instruction must be nonempty text.")
        request = ReflectionRequest(
            image=base.image,
            prompt=task.prompt,
            checklist=task.checklist,
            evaluation_protocol=task.evaluation_protocol or config.DEFAULT_EVALUATION_PROTOCOL,
            reflection_instruction=reflection_instruction,
        )
        return self.reflect(request)

    def _validate_response(
        self, result: ReflectionResult, checklist: Tuple[str, ...]
    ) -> None:
        if not isinstance(result, ReflectionResult):
            raise TypeError("The VLM adapter must return ReflectionResult.")
        if len(result.evaluations) != len(checklist):
            raise ValueError("The VLM must evaluate every checklist item.")
        for question, item in zip(checklist, result.evaluations):
            if item.question != question:
                raise ValueError("VLM evaluations must match the supplied checklist.")
            if item.judgment not in ("pass", "fail", "uncertain"):
                raise ValueError("An item judgment must be pass, fail, or uncertain.")
            score = float(item.score)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("An item score must be in [0, 1].")
            if not isinstance(item.evidence, str):
                raise TypeError("Item evidence must be text.")
        if not isinstance(result.feedback, str) or not result.feedback.strip():
            raise ValueError("feedback must be nonempty text.")
        for name in ("target_text", "reference_text"):
            if not isinstance(getattr(result, name), str):
                raise TypeError(name + " must be text.")
        if bool(result.target_text.strip()) != bool(result.reference_text.strip()):
            raise ValueError("Target and Reference must both be supplied or both be empty.")

# Input preparation and generation-consistency helpers.


def load_task(path, *, evaluation_protocol: Optional[str] = None) -> TaskSpec:
    """Read the prompt and checklist from an external task JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return TaskSpec(
        prompt=data["prompt"],
        checklist=tuple(data["checklist"]),
        evaluation_protocol=evaluation_protocol,
    )


def load_integration(spec: str) -> Mapping[str, Any]:
    """Ask the integration to prepare base, task, and VLM.

    For a supplied record, the factory must resolve compatible native settings
    from that record and attach the loaded flow environment to base.flow.
    When it generates the original image, bind its recorded settings and
    prompt to flow, and save the image, trajectory, seed, and initial
    sampling state in the GenerationRecord.
    """
    module_name, separator, factory_name = spec.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError("integration must have the form module:factory.")
    factory = getattr(importlib.import_module(module_name), factory_name)
    materials = factory(
        flow_model_config=config.FLOW_MODEL_CONFIG,
        seed=config.seed,
        vlm_config=config.VLM_CONFIG,
        input_paths=config.DIR["input"],
    )
    if not isinstance(materials, Mapping):
        raise TypeError("The integration factory must return a mapping of prepared materials.")
    for name in ("base", "task", "vlm"):
        if name not in materials:
            raise ValueError("The integration factory must supply " + name + ".")
    return materials


def load_schedule(spec: str) -> TimeSchedule:
    """Read a supplied s_t(t, T) function for a command-line override."""
    module_name, separator, function_name = spec.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("schedule must have the form module:function.")
    return getattr(importlib.import_module(module_name), function_name)


def _require_matching_settings(
    original: Any,
    current: Any,
    path: str,
    *,
    partial: bool = False,
) -> None:
    """Compare nested native options and identify the first differing field."""
    if partial and original is None:
        return
    if isinstance(original, Mapping):
        if not isinstance(current, Mapping):
            raise ValueError(path + " must be a settings mapping.")
        if any(type(key) is not str for key in original) or any(type(key) is not str for key in current):
            raise TypeError(path + " must use string option names.")
        for key, value in original.items():
            if partial and value is None:
                continue
            name = path + "." + key
            if key not in current:
                source = "original record" if partial else "runtime settings"
                raise ValueError(name + " is missing from the " + source + ".")
            _require_matching_settings(value, current[key], name, partial=partial)
        if not partial:
            extra = set(current) - set(original)
            if extra:
                raise ValueError(path + " has unrecorded settings: " + ", ".join(map(str, extra)))
        return
    if isinstance(original, (tuple, list)):
        if not isinstance(current, (tuple, list)):
            raise ValueError(path + " must be a configuration sequence.")
        if len(original) != len(current):
            raise ValueError(path + ": expected " + str(len(original))
                             + " entries, received " + str(len(current)))
        for index, (left, right) in enumerate(zip(original, current)):
            _require_matching_settings(left, right, path + "[" + str(index) + "]", partial=partial)
        return
    scalar_types = (str, int, float, bool, type(None))
    if type(original) not in scalar_types or type(current) not in scalar_types:
        raise TypeError(path + " must contain ordinary serializable configuration values.")
    matches = original == current
    if isinstance(original, bool) != isinstance(current, bool):
        matches = False
    if not matches:
        labels = ("configured=", ", recorded=") if partial else ("original=", ", supplied=")
        raise ValueError(path + ": " + labels[0] + repr(original) + labels[1] + repr(current))


def check_generation(base: GenerationRecord) -> Tuple[float, ...]:
    """Match the bound original context with the actual prepared environment.

    Check model identity, native settings, sampling start seed, configured choices,
    and planned times before requesting reflection or executing native sampling.
    """
    flow = base.flow
    for reader in ("generation_model_id", "generation_settings", "sampling_times"):
        if not callable(getattr(flow, reader, None)):
            raise TypeError("The flow integration must implement " + reader + "().")
    runtime_model_id = flow.generation_model_id()
    _require_matching_settings(flow.model_id, runtime_model_id, "model_id")

    times = tuple(float(point.time) for point in base.trajectory)
    if len(times) < 2 or any(not math.isfinite(time) for time in times):
        raise ValueError("The trajectory must contain at least two finite times.")
    deltas = tuple(right - left for left, right in zip(times, times[1:]))
    if not (all(delta > 0.0 for delta in deltas) or all(delta < 0.0 for delta in deltas)):
        raise ValueError("Trajectory times must be strictly monotonic.")
    if not isinstance(flow.model_settings, Mapping):
        raise TypeError("flow.model_settings must contain the original native options.")
    if "seed" in flow.model_settings:
        raise ValueError("Store the initialization seed in base.sampling_state, not flow.model_settings.")
    num_steps = flow.model_settings.get("num_steps")
    if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps < 1:
        raise ValueError("flow.model_settings.num_steps must be a positive integer.")
    if num_steps != len(times) - 1:
        raise ValueError("num_steps: original settings specify " + str(num_steps)
                         + " updates, but the trajectory records " + str(len(times) - 1))

    if base.sampling_state is not None and not isinstance(base.sampling_state, Mapping):
        raise TypeError("base.sampling_state must contain the seed and native start state.")
    original_settings = dict(flow.model_settings)
    if base.sampling_state is not None and "seed" in base.sampling_state:
        original_settings["seed"] = base.sampling_state["seed"]
    runtime_settings = flow.generation_settings()
    _require_matching_settings(original_settings, runtime_settings, "model_settings")
    runtime_steps = runtime_settings["num_steps"]
    if type(runtime_steps) is not int or runtime_steps < 1:
        raise ValueError("runtime model_settings.num_steps must be a positive integer.")
    runtime_times = tuple(float(time) for time in flow.sampling_times())
    _require_matching_settings(times, runtime_times, "time_grid")

    choices = config.FLOW_MODEL_CONFIG
    model_id = choices.get("model_id")
    if model_id is not None:
        _require_matching_settings(model_id, flow.model_id, "config.model_id", partial=True)
    requested_steps = choices.get("num_steps")
    if requested_steps is not None and (type(requested_steps) is not int or requested_steps < 1):
        raise ValueError("config.num_steps must be a positive integer.")
    if requested_steps is not None:
        _require_matching_settings({"num_steps": requested_steps}, original_settings, "config", partial=True)
    if config.seed is not None:
        _require_matching_settings({"seed": config.seed}, original_settings, "config", partial=True)
    _require_matching_settings(choices.get("settings", {}), original_settings, "config.settings", partial=True)
    return times
