# Simulate Ca Video

This project provides a MATLAB function `Simulate_Ca_video` to simulate calcium imaging videos with various parameters and options. This documentation will guide you through the usage, parameters, and inner workings of the function.

## Quick Start

To get started, you can call the function with default or customized parameters:

```matlab
[Mot, outpath, file_name, A, GT_motion_xy] = Simulate_Ca_video('Nneu', 100, 'ses', 2, 'CA1_A', true, 'PNR', 1);
```

The final output `GT_motion_xy` stores per-session x/y displacements applied within each session and is also written to the metadata file when saving results.

### Coordinate frame of the saved ground truth

`Add_NRmotion` removes 20 px from every border of the warped data, so the saved
videos are 40 px smaller than the requested `d` in each dimension (`d = [440 600]`
produces a 400x560 video). The ground-truth footprints `A_GT` and `bA_GT` are
cropped to that same field of view before being written to `*_meta.mat`, so
`A_GT`, `A`, the session videos and the GT video all share one coordinate frame
and `A_GT{s}` can be compared against extracted footprints without any manual
offset. Note that `d` is `[rows, cols]`, i.e. height first.

### Spatially heterogeneous density

By default neurons are scattered uniformly (subject to `min_dist`) inside the
heart-shaped mask. `dense_region` adds one or more rectangles that receive a
different density, so the field of view has dense clumps and sparse stretches:

```matlab
Simulate_Ca_video('d',[440 600], 'Nneu',150, ...
                  'min_dist',14, ...                       % separation outside
                  'dense_region',[135 266 141 280], ...    % saved-video coords
                  'dense_frac',0.7, ...                    % 70% of neurons inside
                  'dense_min_dist',5);                     % separation inside
```

Rules:

- **Coordinates.** `dense_region` rows are `[r1 r2 c1 c2]` in **saved-video**
  pixels by default, i.e. the frame written to disk, which is 40 px smaller than
  `d` on each axis. The +20 px offset into the simulation frame is applied
  internally. Pass `'dense_region_frame','full'` to give the rectangle in `d`
  coordinates instead.
- **Order.** Rectangles are filled first, then the background gets the
  remainder (`Nneu - sum(dense counts)`). Overlapping rectangles are made
  mutually exclusive: a pixel belongs to the first rectangle that claims it.
- **Separation.** A candidate point must clear the minimum separation *of its own
  region* against **every** point already placed, including points from other
  regions. That is what keeps a dense clump and the sparse background from
  piling onto each other along the rectangle border. It also means a band
  roughly `min_dist` wide just outside a rectangle is thinner than the rest of
  the background.
- **Mask.** Rectangles are intersected with the heart mask (or with `ones(d)`
  when `create_mask=false`), so the mask is always respected.
- **Under-filling.** If a region cannot hold the requested number of points at
  the requested separation, a warning is issued and fewer points are placed.
- **Bookkeeping.** `*_meta.mat` now also contains a `placement` struct with the
  neuron coordinates in both frames (`X`, `Y` in `d` coordinates; `X_saved`,
  `Y_saved` in saved-video coordinates), the `region_id` of every neuron
  (`1..M` for the rectangles, `M+1` for the background), the rectangles in both
  frames, and the per-region counts and separations. It is written for
  homogeneous runs too.

`placement_margin` forbids neuron centres within N pixels of the **full-frame**
border. It only matters with `create_mask=false`: without it, neurons can land in
the 20 px strip that `Add_NRmotion` crops away, and their ground-truth footprints
come out clipped or empty. Neuron kernels are 50x50, so `'placement_margin',50`
is a safe value. With the heart mask on (the default), placement is already far
from the border and this option changes nothing.

Leaving `dense_region` empty and `placement_margin` empty reproduces the original
placement code path exactly, bit for bit, for a given seed.

### Random regions with spatially coherent activity

`region_areas` replaces the rectangles with a partition of the field of view into
random, contiguous, irregularly shaped regions, and additionally gives each
region its own **activity coherence**: inside a region, neurons that sit close
together fire together more often than chance.

```matlab
Simulate_Ca_video('d',[360 360],'F',400,'ses',1, ...
                  'region_areas',[25600 25600 25600 25600], ...  % saved-frame px
                  'region_n',[400 400 400 400], ...
                  'region_min_dist',[2.5 2.5 2.5 2.5], ...
                  'region_corr',[0 0.3 0.6 0.9], ...
                  'corr_length',3, ...                            % in gSig units
                  'region_seed',7, ...
                  'spike_prob',[log(0.5) 2.25], ...
                  'create_mask',false,'outpath',pwd);
```

- **Layout.** `grow_random_regions` grows the regions on the **saved-frame**
  geometry (`d` minus 40 on each axis), so `region_areas` is in saved-video
  pixels, the same convention as `'dense_region_frame','saved'`. The masks are
  shifted by 20 px into the full frame, which means no neuron is ever placed in
  the strip that `Add_NRmotion` crops away. Because the regions must jointly
  cover the frame, only the **ratios** between the entries are honoured.
- **Layout seed.** `region_seed` seeds the layout only, and the global random
  state is restored afterwards, so changing it does not change anything else.
  `[]` falls back to `seed`.
- **Coherence.** Every neuron inherits the `region_corr` of the region it was
  placed in, and the spike trains come from `correlated_spikes` instead of the
  independent draw in `create_random_spike`. The latent pairwise correlation is
  `region_corr * exp(-distance / (corr_length * gSig))`, with `gSig` measured
  from the neuron footprints actually drawn (about 2.7 px at `scale` 1).
- **Firing rate matters a lot.** Thresholding a Gaussian into a binary spike
  train destroys most of the latent correlation when spikes are rare. At the
  default `spike_prob` of `[-4.9 2.25]` the median per-frame probability is
  7.5e-4, i.e. under one spike per neuron in 400 frames, and the achieved
  correlation is indistinguishable from zero whatever `region_corr` is. Raise
  `spike_prob(1)` (e.g. `log(0.5)`, a median rate of 0.05 per frame at `sf` 10)
  when the coherence is meant to be measurable.
- **Measured, not assumed.** The metadata file stores a `region_info` struct with
  the per-neuron `region_id` and `corr_strength`, the region masks in saved-frame
  geometry, `corr_length` (and `corr_length_px`), the pairwise neuron distances,
  and the **achieved pairwise calcium trace correlation matrix** `trace_corr`
  (plus `trace_corr_per_session`). The trace correlation is not the requested
  `region_corr` and not the spike correlation, so it is measured every run.

`region_info` is `[]` and nothing changes when `region_areas` is left empty.

For detailed information on each parameter and sub-function, please refer to the [Function Documentation](#function-documentation) section.

## Function Documentation

### Syntax

```matlab
[out, Mot, outpath, file_name, A] = Simulate_Ca_video(varargin)
```

### Description

The `Simulate_Ca_video` function generates simulated calcium imaging videos with neural data, across-session misalignment artifacts, within-session motion, and noise. It allows for customization of various parameters including the number of neurons, sessions, frame rate, and more.

### Parameters

- **'outpath'**: Output path where simulation results will be saved.
- **'min_dist'**: Minimum distance between neurons in the simulated data.
- **'Nneu'**: Number of neurons to be simulated.
- **'dense_region'**: `M x 4` matrix, one `[r1 r2 c1 c2]` rectangle per row (inclusive pixel indices), giving regions of raised neuron density. `[]` (default) keeps the original homogeneous placement. See *Spatially heterogeneous density* below.
- **'dense_frac'**: Fraction of `Nneu` placed inside the rectangles. Scalar = total dense fraction, split evenly between rectangles; vector = one fraction per rectangle. Default `0.5`.
- **'dense_min_dist'**: Minimum neuron separation inside the rectangles (scalar, or one per rectangle). `[]` (default) means "same as `min_dist`", i.e. no density contrast.
- **'dense_region_frame'**: `'saved'` (default) or `'full'`, the coordinate frame `dense_region` is expressed in.
- **'placement_margin'**: Forbid neuron centres within this many pixels of the **full-frame** (`d`) border. `[]` (default) keeps the original behaviour.
- **'region_areas'**: `1 x R` target region areas in **saved-frame** pixels. Non-empty switches on the random-region layout and the spatially coherent spike generator. `[]` (default) keeps the original behaviour. Only the ratios between the entries are honoured. See *Random regions with spatially coherent activity* below.
- **'region_n'**: `1 x R` number of neurons per region. `[]` (default) splits `Nneu` in proportion to the area available in each region.
- **'region_min_dist'**: `1 x R` minimum neuron separation per region, in pixels. `[]` (default) uses `min_dist` everywhere.
- **'region_corr'**: `1 x R` zero-distance activity correlation of each region, in `[0, 1)`. `[]` (default) means zero, i.e. no coherence.
- **'corr_length'**: Correlation decay length, in `gSig` units. Default `3`.
- **'region_seed'**: Seed for the region layout only. `[]` (default) falls back to `seed`. The global random state is restored afterwards.
- **'PNR'**: Peak-to-Noise Ratio for the simulated data.
- **'d'**: Dimensions of the simulated video frames (width and height) in pixels.
- **'F'**: Frame rate (frames per second) for the simulated calcium video data.
- **'overlap'**: Proportion of neurons remapping across multiple sessions.

Two separate things move the image, and every option name says which one.
`across_session_*` is one fixed warp per session, which is what the inter-session
**alignment** stage has to undo. `within_session_*` is a different shift on every
frame, which is what **motion correction** has to undo. They are independent: a
session can be misaligned without shaking, shake without being misaligned, or both.

Across-session misalignment:

- **'across_session_nonrigid'**: Amplitude of the smooth warp between sessions, in pixels. `0` (default) means none.
- **'across_session_nonrigid_scale'**: How broad that warp is, as the Gaussian sigma in pixels used to smooth the random field. Default `60`. Large values bend the frame in one slow sweep, small values ripple it.
- **'across_session_translation'**: Length in pixels of the rigid shift between sessions. Default `1`. The direction is random, the length is this.

Within-session motion (an AR(1) walk, so consecutive frames move together):

- **'within_session_translation_std'**: Standard deviation in pixels of the whole-frame shake. Default `3`. A scalar applies to every session, or give one value per session so they differ.
- **'within_session_nonrigid_std'**: Standard deviation in pixels of the bend applied on top of that shake. `0` (default) keeps the motion purely rigid.
- **'within_session_nonrigid_grid'**: Control points per axis of that bend. Default `3`, i.e. a 3x3 grid.
- **'within_session_motion_smoothness'**: AR(1) coefficient, shared by the shake and the bend. Default `0.99`. Near 1 gives slow drift, near 0 gives frame-to-frame jitter.
- **'within_session_motion_max'**: Optional cap on peak displacement, in pixels.
- **'within_session_motion_seed'**: Seed offset, for a reproducible trajectory.

Within-session motion that changes character over time, all off by default
(`make_motion_regimes`; drawn from a stream of its own, seed + 3000 + session, so
switching it on changes nothing else in the recording):

- **'within_session_rigid_frames'**: `[min max]` length in frames of rigid-only stretches. `[]` (default) = no regimes, the bend is on throughout. Rigid and bend stretches alternate; the bends (spline and local) are multiplied by a weight that is 0 in a rigid stretch and 1 in a bend stretch.
- **'within_session_bend_frames'**: `[min max]` length of the stretches with bends. Default `[100 300]`.
- **'within_session_regime_ramp'**: Frames over which the bends fade in and out at a regime change. Default `10`.
- **'within_session_local_bends'**: Average number of local bends under way at once inside a bend stretch. `0` (default) = none. A local bend is a Gaussian bump of displacement at a random place that fades in and out over its lifetime; half are *shift* bumps (a patch slides one way), half *swell* bumps (it pushes out from, or pulls into, a centre, so the patch as a whole barely moves).
- **'within_session_local_bend_width'**: `[min max]` Gaussian sigma in pixels, log-uniform. Default `[10 40]`.
- **'within_session_local_bend_amp'**: `[min max]` peak displacement in pixels. Default `[0.5 3]`.
- **'within_session_local_bend_life'**: `[min max]` lifetime in frames. Default `[40 200]`.
- **'within_session_local_bend_strain'**: Cap on a bump's peak strain (largest gradient of its displacement), which caps the amplitude of the narrow ones. Default `0.1`.
- **'within_session_local_bend_inset'**: Keep bump centres this many pixels from the frame border. Default `0`.
- **'within_session_jumps'**: Sudden rigid jumps per 1000 frames. `0` (default) = none.
- **'within_session_jump_size'**: `[min max]` jump length in pixels. Default `[5 15]`.
- **'within_session_jump_held'**: Fraction of jumps that stay at the new place; the others decay back exponentially. Default `0.5`.
- **'within_session_jump_decay'**: `[min max]` decay time constant in frames. Default `[10 40]`.
- **'within_session_jump_bound'**: No jump lands farther than this many pixels from the start. Default `15`.
- **'within_session_jump_gap'**: Minimum frames between two jumps. Default `100`.

Camera model, off by default:

- **'camera_supersample'**: `s > 1` records every frame as a camera does. The scene is the band-limited interpolant of the simulated frame (`dct_interp_matrix`), the within-session motion moves it, and each pixel averages it at the centres of `s x s` sub-pixels (`camera_setup`, `apply_camera_motion`). The default path moves frames by bilinear interpolation, which blurs each frame by an amount that depends on its fractional shift and pulls any correlation engine about 0.1 px toward whole pixels; this path does not (the pull falls below 0.005 px). Odd `s` keeps the middle sub-pixel on the pixel centre. `1` (default) = off.

With any of these on, every frame takes one path: the shake, the spline bend, the
regimes, the local bends and the jumps are composed into one field
(`within_session_field`) and applied once, with no intermediate `v2uint8`, so a
frame keeps the grey levels of the still scene. The motion-free twin is the same
call with the motion switched off and `camera_supersample` kept. The draws are
saved in `*_meta.mat` as `GT_regimes{session}` (`bend_weight`, `stretches`,
`bumps`, `jumps`, `jump_xy`, `camera_supersample`). With all of them off, the
recording is the same as before, bit for bit, for a given seed.

Noise, both off by default:

- **'sensor_noise'**: Standard deviation, in grey levels, of white Gaussian noise added to every recorded frame after the motion: independent per pixel and per frame, rounded and clipped to `uint8`. With the camera model it goes in before the camera rounds the frame, as a sensor is digitised once: rounded first, a smooth noise-free scene becomes flat steps whose edges move in whole-pixel steps, which pulls registration toward whole pixels (about 0.08 px), and noise added afterwards does not undo it. Without the camera model it is added after the last warp. It is drawn from a stream of its own (seed + 4000 + session), so it does not move with the tissue, a motion-free twin gets the same draw, and the rest of the recording is unchanged. `0` (default) = none.
- **'scene_noise'**: `false` leaves the per-frame noise out of the scene. The noise is still drawn and is replaced by its mean, so the frame keeps the grey scale it would have had and every other draw is unchanged. `true` (default) = as always: the noise is part of the scene, moves with the tissue and, with the camera model on, is averaged by it. `'scene_noise', false, 'sensor_noise', s` moves the noise from the scene to the sensor; with both off, the frames are noise-free.

- **'ses'**: Number of sessions to be generated.
- **'seed'**: Random number generator seed.
- **'B'**: Baseline id for different baseline images.
- **'spike_prob'**: Probability distribution parameters for spike events.
- **'save_videos'**: Flag to save session videos (.avi/.h5).
- **'save_mat'**: Flag to save the metadata `.mat` file.
- **'save_GT'**: Flag to save the ground-truth video.
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
[Mot, outpath, file_name, A, GT_motion_xy] = Simulate_Ca_video('Nneu', 100, 'ses', 2, 'CA1_A', true, 'PNR', 1);

% Example 2
[Mot, outpath, file_name, A, GT_motion_xy] = Simulate_Ca_video('Nneu', 50, 'ses', 1, 'F', 18000, 'LFP', 8, 'spike_prob', [-4.91, 0.83], 'sf', 60);

% Example 3
[Mot, outpath, file_name, A, GT_motion_xy] = Simulate_Ca_video('Nneu', 300, 'ses', 20, 'F', 1500, 'across_session_nonrigid', 0, 'min_dist', 2, 'spike_prob', [-2, 0.83], 'A2', true, 'overlap', 0.2);
```

### Detailed Function Workflow

1. **Initialize Variables**: Using `int_var` to parse input parameters.
2. **Create Baselines**: Generate baseline images using `create_baseline`.
3. **Create Neural Data**: Generate neural data using `create_neuron_data`.
4. **Apply Overlapping Mask**: Apply remapping and drifting effects to the data.
5. **Add Non-Rigid Misalignment**: Introduce across-session misalignment artifacts to the sessions using `Add_NRmotion`.
6. **Create Noise Data**: Generate and add noise to the simulated data.
7. **Apply Within-Session Motion**: Use AR-driven translations to inject realistic frame-to-frame motion into each session.
8. **Integrate Model**: Combine all components to form the final simulated video.
9. **Save Results**: Save the output according to `save_videos`, `save_mat`, and `save_GT` (metadata now includes within-session motion ground truth).

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
Adds non-rigid misalignment artifacts across sessions.

### saveash5
Saves data in HDF5 format.
