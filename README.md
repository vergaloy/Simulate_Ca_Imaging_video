# Simulate Ca Video

This project provides a MATLAB function `Simulate_Ca_video` to simulate calcium imaging videos with various parameters and options. This documentation will guide you through the usage, parameters, and inner workings of the function.

## Quick Start

To get started, you can call the function with default or customized parameters:

```matlab
[out, Mot, outpath, file_name, A] = Simulate_Ca_video('Nneu', 100, 'ses', 2, 'CA1_A', true, 'PNR', 1);
```

For detailed information on each parameter and sub-function, please refer to the [Function Documentation](#function-documentation) section.

## Function Documentation

### Syntax

```matlab
[out, Mot, outpath, file_name, A] = Simulate_Ca_video(varargin)
```

### Description

The `Simulate_Ca_video` function generates simulated calcium imaging videos with neural data, motion artifacts, and noise. It allows for customization of various parameters including the number of neurons, sessions, frame rate, and more.

### Parameters

- **'outpath'**: Output path where simulation results will be saved.
- **'min_dist'**: Minimum distance between neurons in the simulated data.
- **'Nneu'**: Number of neurons to be simulated.
- **'PNR'**: Peak-to-Noise Ratio for the simulated data.
- **'d'**: Dimensions of the simulated video frames (width and height) in pixels.
- **'F'**: Frame rate (frames per second) for the simulated calcium video data.
- **'overlap'**: Proportion of neurons remapping across multiple sessions.
- **'motion'**: Inter-session misalignment amplitude.
- **'motion_sz'**: Size of the motion effect applied to the frames.
- **'translation'**: Add translation misalignment in addition to Non-rigid motion.
- **'ses'**: Number of sessions to be generated.
- **'seed'**: Random number generator seed.
- **'B'**: Baseline id for different baseline images.
- **'spike_prob'**: Probability distribution parameters for spike events.
- **'save_files'**: Flag to save simulation results.
- **'create_mask'**: Flag to create a mask during the simulation.
- **'comb_fact'**: Combination factor used in the simulation process.
- **'drift'**: Presence of drifting activities in the simulated data.
- **'LFP'**: Flag to simulate Local Field Potentials data.
- **'sf'**: Scaling factor used in the simulation process.
- **'plotme'**: Flag to generate plots during simulation.
- **'invtRise'**: Parameters related to the rising phase of calcium signals.
- **'invtDecay'**: Parameters related to the decaying phase of calcium signals.
- **'disappearBV'**: Erode baseline value.
- **'CA1_A'**: Change neurons shapes across sessions using CA1 data.
- **'A2'**: Use CA1 neurons shapes instead of DG.
- **'force_active'**: Ensure all neurons have at least one Calcium transient per session.

### Usage Examples

```matlab
% Example 1
[out, Mot, outpath, file_name, A] = Simulate_Ca_video('Nneu', 100, 'ses', 2, 'CA1_A', true, 'PNR', 1);

% Example 2
[out, Mot, outpath, file_name, A] = Simulate_Ca_video('Nneu', 50, 'ses', 1, 'F', 18000, 'LFP', 8, 'spike_prob', [-4.91, 0.83], 'sf', 60);

% Example 3
[out, Mot, outpath, file_name, A] = Simulate_Ca_video('Nneu', 300, 'ses', 20, 'F', 1500, 'motion', 0, 'min_dist', 2, 'spike_prob', [-2, 0.83], 'A2', true, 'overlap', 0.2);
```

### Detailed Function Workflow

1. **Initialize Variables**: Using `int_var` to parse input parameters.
2. **Create Baselines**: Generate baseline images using `create_baseline`.
3. **Create Neural Data**: Generate neural data using `create_neuron_data`.
4. **Apply Overlapping Mask**: Apply remapping and drifting effects to the data.
5. **Add Non-Rigid Motion**: Introduce motion artifacts to the sessions using `Add_NRmotion`.
6. **Create Noise Data**: Generate and add noise to the simulated data.
7. **Integrate Model**: Combine all components to form the final simulated video.
8. **Save Results**: Save the output if the `save_files` flag is set.

## Sub-Functions

### int_var
Parses and initializes input parameters for the simulation.

### get_out_path
Determines the output path for saving simulation results.

### create_baseline
Generates baseline images for the simulation.

### add_remapping_drifting
Applies remapping and drifting effects to the neural data.

### Add_NRmotion
Adds non-rigid motion artifacts to the sessions.

### saveash5
Saves data in HDF5 format.