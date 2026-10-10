# Reflection-Guided Dynamics Rectification (RGDR)

Reflection-Guided Dynamics Rectification (RGDR) is a training-free, trajectory-level method for correcting residual errors in existing image generations. It addresses the gap between identifying an error in a completed image and making the generative model carry out the intended correction.

RGDR turns a vision-language model's reflection into two descriptions: **Target** describes the desired corrected content, and **Reference** describes the corresponding erroneous content in the original image. The difference between their conditional velocity predictions defines a correction field. At each sampling step, this field is scaled relative to the model's native base velocity and added to guide a new trajectory from the original initial state. Correction strength and temporal weighting are adjustable, while model parameters remain fixed.

This repository provides a general framework for connecting RGDR to a chosen flow model and vision-language model and performing one complete rectification.

## Files

- **config.py** contains the model choices, initialization seed, file locations, RGDR parameters, and instructions for the vision-language model.
- **utils.py** defines the data formats and model interfaces. It also provides tools for reading tasks and checking that the model environment matches the original generation.
- **RGDR.py** implements the rectification method and the complete procedure for generating one corrected image.
- **scripts/test_rgdr.py** prepares the inputs, runs RGDR, and saves the result.

## Configuration

Set the following options in **config.py**:

- **Flow model:** choose the model, number of sampling steps, and native generation settings, including resolution, sampling method, and guidance.
- **Initialization seed:** specify the seed used to generate the original image.
- **Vision-language model:** choose the model and its supported loading and request options.
- **File locations:** specify the external preparation function, input files, and output directory in **DIR**.
- **RGDR parameters:** set the correction strength **eta**, normalization stabilizer **epsilon**, and temporal weighting function **s_t**.
- **VLM instructions:** specify how the image should be evaluated and how the correction should be described.

When using an existing image, the selected model and generation settings must match its original generation record.

## Usage

### 1. Prepare the original generation

Provide the original image, its complete sampling trajectory, the generation prompt, and the checklist.

The generation record must preserve the initial state and the information needed to restart the original sampling process.

### 2. Connect the model and VLM

Prepare the chosen models in your external integration code.

For the flow model, connect its native velocity calculation, text encoding, sampling process, and image decoding to the interfaces in **utils.py**. The native velocity must include the processing normally used by that model.

For the vision-language model, connect a call that evaluates the image and returns the checklist scores, feedback, and paired correction descriptions. **Target** describes the desired corrected content. **Reference** describes the corresponding erroneous content in the original image.

The preparation function receives the flow model configuration, seed, VLM configuration, and input paths. It returns **base**, **task**, and **vlm** in the formats defined in **utils.py**. To save the generated image and trajectory, also provide a **save_result** function.

### 3. Run RGDR

Set the preparation function location in **DIR**, using the `module:function` format. Ensure that the module is available in your Python environment.

Run the following command from the repository root:

```bash
python scripts/test_rgdr.py
```

RGDR evaluates the original image, computes the correction, and uses the original model's sampling process to generate a new image and trajectory. The result retains the original generation record and the RGDR parameters used for the correction.
