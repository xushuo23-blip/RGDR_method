"""Reflection-Guided Dynamics Rectification (RGDR).

Build the rectified native velocity field and perform one complete correction.
"""

import math
from typing import Optional

import config
from utils import (
    GenerationRecord, ReflectionResult, RGDRResult, RGDRSettings,
    State, TaskSpec, TimeSchedule, TrajectoryPoint, VelocityField, VLMAdapter,
    check_generation,
)


def make_rectified_field(
    base: GenerationRecord,
    reflection: ReflectionResult,
    settings: RGDRSettings,
) -> VelocityField:
    """Encode the fixed correction texts and build the RGDR field.

    D = V_target - V_reference
    D_bar = ||V_base|| D / (||D|| + epsilon)
    V_RGDR = V_base + eta s_t(t, T) D_bar
    """
    flow = base.flow
    T = float(base.trajectory[0].time)
    has_correction = bool(reflection.target_text.strip())
    if has_correction != bool(reflection.reference_text.strip()):
        raise ValueError("Target and Reference must both be supplied or both be empty.")
    target = flow.encode_condition(reflection.target_text) if has_correction else None
    reference = flow.encode_condition(reflection.reference_text) if has_correction else None

    def velocity(t: float, state: State) -> State:
        if not math.isfinite(float(t)):
            raise ValueError("The current model time must be finite.")

        base_velocity = flow.base_velocity(state, t)
        if not has_correction:
            return base_velocity
        weight = float(settings.s_t(float(t), T))
        coefficient = settings.eta * weight
        if not math.isfinite(weight) or weight < 0.0 or not math.isfinite(coefficient):
            raise ValueError("s_t must return a finite nonnegative weight with finite eta * s_t.")
        if coefficient == 0.0:
            return base_velocity

        target_velocity = flow.conditional_velocity(state, t, target)
        reference_velocity = flow.conditional_velocity(
            state, t, reference
        )
        direction = target_velocity - reference_velocity
        base_norm = float(flow.norm(base_velocity))
        direction_norm = float(flow.norm(direction))
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (base_norm, direction_norm)
        ):
            raise ValueError("Velocity norms must be finite and nonnegative.")

        normalized_direction = (base_norm / (direction_norm + settings.epsilon)) * direction
        return base_velocity + coefficient * normalized_direction

    return velocity


def run_rgdr(
    base: GenerationRecord,
    task: TaskSpec,
    vlm: VLMAdapter,
    *,
    eta: Optional[float] = None,
    s_t: Optional[TimeSchedule] = None,
    epsilon: Optional[float] = None,
    reflection_instruction: Optional[str] = None,
) -> RGDRResult:
    """Connect the supplied environment and inputs to one complete RGDR pass."""
    # 1. Check the shared native generation settings and resolve RGDR parameters.
    check_generation(base)
    eta = float(config.eta if eta is None else eta)
    epsilon = float(config.epsilon if epsilon is None else epsilon)
    s_t = config.s_t if s_t is None else s_t
    if not math.isfinite(eta) or eta < 0.0:
        raise ValueError("eta must be finite and nonnegative.")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive.")
    if not callable(s_t):
        raise TypeError("s_t must be a function of the current time t and initial time T.")
    settings = RGDRSettings(
        eta=eta,
        epsilon=epsilon,
        s_t=s_t,
    )

    # 2. Evaluate the original image and obtain its four-part reflection.
    reflection = vlm.reflect_image(
        base, task,
        reflection_instruction=reflection_instruction,
    )

    # 3. Supply the RGDR velocity to the model's native sampling process.
    field = make_rectified_field(
        base, reflection, settings
    )

    # 4. Generate the new trajectory and decode its endpoint.
    flow = base.flow
    initial_state = flow.copy_state(base.trajectory[0].state)
    flow.restore_sampling_state(base.sampling_state)
    time_grid = check_generation(base)
    accepted = tuple(flow.integrate(field, initial_state, time_grid))
    if (
        len(accepted) != len(time_grid)
        or tuple(float(point.time) for point in accepted) != time_grid
    ):
        raise ValueError("The sampler must return all accepted native time boundaries.")
    trajectory = tuple(
        TrajectoryPoint(float(point.time), flow.copy_state(point.state))
        for point in accepted
    )
    image = flow.decode(flow.copy_state(trajectory[-1].state))
    return RGDRResult(image=image, trajectory=trajectory, base=base, settings=settings)
