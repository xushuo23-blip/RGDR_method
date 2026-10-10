# Reflection-Guided Dynamics Rectification (RGDR)

RGDR evaluates an existing image against its prompt and checklist, obtains Target
and Reference descriptions from a VLM, and generates one corrected image from the
original initial state. The supplied flow model provides its complete native
velocity and runs its original sampling process with the RGDR correction.

This is a generic integration framework. Connect your flow model and VLM using
their native environments, weights, and dependencies.

## Files

- `config.py`: model choices, seed, paths, RGDR parameters, and VLM instructions.
- `utils.py`: data classes, `FlowModel`, `VLMAdapter`, and preparation helpers.
- `RGDR.py`: the correction formula and `run_rgdr` for one complete pass.
- `scripts/test_rgdr.py`: prepare inputs, run RGDR, and save the result.

## Configuration

Edit `config.py`:

- `FLOW_MODEL_CONFIG`: model identifier, step count, and other native options in
  `settings`, such as resolution, sampler, and guidance.
- `seed`: the initialization seed, recorded in `base.sampling_state`.
- `VLM_CONFIG`: VLM identifier and supported loading/request options.
- `DIR`: `integration` locates the external preparation function; `input` contains
  the task, image, and generation-record paths; `output` is the result directory.
- `eta`, `epsilon`, `s_t(t, T)`: correction strength, normalization stabilizer,
  and temporal weight. Here `t` is native model time and `T` is the first
  trajectory time.
- `DEFAULT_EVALUATION_PROTOCOL`, `DEFAULT_REFLECTION_INSTRUCTION`: the two VLM
  instructions for evaluation and correction.

For an existing original image, use its recorded model, settings, and sampling
start. Explicit configuration choices must match that record. The step count
and native sampling options determine the model's time grid.

## Connect the model and VLM

Create `flow` from the original model context and its connected operations.
Here `native` denotes your prepared native environment; adapt its operations to
the signatures defined in `utils.py` and bind them to the original context.

```python
flow = FlowModel(
    model_id=original_model_id,
    model_settings=original_model_settings,
    prompt=original_prompt,
    generation_model_id=native.generation_model_id,
    generation_settings=native.generation_settings,
    sampling_times=native.sampling_times,
    base_velocity=native.base_velocity,
    encode_condition=native.encode_condition,
    conditional_velocity=native.conditional_velocity,
    norm=native.norm,
    copy_state=native.copy_state,
    restore_sampling_state=native.restore_sampling_state,
    integrate=native.integrate,
    decode=native.decode,
)

base = GenerationRecord(
    flow=flow,
    image=original_image,
    trajectory=original_trajectory,
    sampling_state=initial_sampling_state,
)
vlm = VLMAdapter(call_model=connected_vlm_call)
```

Each trajectory point stores one native time and an independent state snapshot.
Include the initial state and every accepted boundary: 50 updates produce 51
points. Save any required RNG/sampler start state after latent initialization
and before sampling preparation. RGDR restores that state and reuses the recorded
initial latent.

The runtime readers independently report the actual model, options, and time grid
for consistency checks. Velocity predictions share the same state representation,
time, sign, shape, and units. `integrate` calls the original pipeline or solver
with the supplied velocity at its point of use. Keep inference parameters frozen
and disable gradient recording in the connected model environment.

Prepare `task` from its prompt and checklist, using `load_task` or `TaskSpec`.
`connected_vlm_call` receives `ReflectionRequest` and returns `ReflectionResult`
with per-item evaluations, feedback, `target_text`, and `reference_text`. Each
item contains its question, judgment, fulfillment score in [0, 1], and evidence.
The program computes the mean score times 100. Target describes the desired
correction; Reference describes the same erroneous content in the original image.
When no correction is needed, both texts are empty.

## Run one correction

Set `DIR["integration"]` to your importable preparation function, for example
`"your_project.integration:prepare"`. It receives `flow_model_config`, `seed`,
`vlm_config`, and `input_paths`, and returns a mapping containing `base`, `task`,
and `vlm`. When `DIR["output"]` is set, also return a
`save_result(result, output_directory)` function for your image and state formats.

From the repository root, with that preparation module available to Python, run:

```sh
python scripts/test_rgdr.py
```

With materials already prepared, call the core directly:

```python
result = run_rgdr(base=base, task=task, vlm=vlm)
new_image = result.image
new_trajectory = result.trajectory
```

`RGDRResult` retains the new image and trajectory, original `base`, and additional
RGDR `settings`. Each sampling evaluation applies:

```text
D = V_target(t, x) - V_reference(t, x)
D_bar = norm(V_base(t, x)) * D / (norm(D) + epsilon)
V_RGDR = V_base(t, x) + eta * s_t(t, T) * D_bar
```
