function R = real_ranges(F, seed, fps)
%% The 'real' defaults (S28): every condition at the real range, x 1.2.
%
% Pablo, 2026-11-12: "the range of simulated condition [is] 1.2 the real
% range observed". The real ranges are the control recordings' (PatchForge
% docs/lanes/s28-sim-real-ranges.md: 48 sessions of 6 mice, 648 x 486, 8 bit,
% 20 Hz; measured in S4, S24, S25 and S28). The rule: a range runs from the
% real minimum / 1.2 to the real maximum x 1.2; a lower end of 0 stays 0.
%
% Two kinds of default:
%   * a range, where only a real range is known: drawn by the option's own
%     stream as before (uniform, or log-uniform for sizes and rates);
%   * a real session, where the recordings were measured one by one (the
%     tables in real_sessions_control.mat): each movie draws ONE session per
%     table and takes its values, each times its own factor drawn
%     log-uniformly in [1/1.2, 1.2]. The values then span exactly the rule's
%     range, a real 0 stays 0, and the shares follow the real ones (a third of
%     the movies get the quiet sessions' low noise and low fixed pattern, and
%     with them the strong rings; two thirds get no glitches). The camera's
%     traits (noise, black level, fixed pattern, exposure) come from one
%     session together, as they belong to one camera and one preparation.
%
% Inputs: F frames, seed (the movie's), fps (the clock for every rate: 20,
% the recordings'). All draws come from a stream of their own, seed + 8950,
% in a fixed order, so every run of one movie (moving, twins, noise-free)
% gets the same values.
%
% Output R:
%   nv       - every default as name/value pairs (what Simulate_Ca_video
%              fills in for the options the caller left out)
%   scene    - the scene's (all runs): smooth background, scene noise off,
%              the S25 peaks (population events with the neuropil glow)
%   camera   - fixed to the camera (the camera runs): read-out, pattern, hot
%              pixels, vignetting, flicker, trend, exposure
%   moving   - the moving run's only: slow drift, jolts, glitches
%   motion   - the fast motion at the real range (shake, broad bends; no
%              regimes or jumps): the driver's motion setting 'real'
%   noise    - the sensor noise (grey, at the simulated pixel)
%   draw     - which session each table gave and its factors
%
% Author: Pablo Vergara

if nargin < 3 || isempty(fps), fps = 20; end
here = fileparts(mfilename('fullpath'));
Tb = load(fullfile(here, 'real_sessions_control.mat'));
rs = RandStream('mt19937ar', 'Seed', seed + 8950);
minutes = F / (fps * 60);
draw = struct();

%% The camera: one session for noise, black level, fixed pattern, exposure
[c, draw.camera] = session(rs, Tb.camera, {'sensor_noise', 'black_level', 'sensor_dsnu', ...
    'sensor_prnu', 'sensor_row_offset', 'sensor_col_offset', 'exposure_target'});
[h, draw.hot] = session(rs, Tb.hot, {'hot_pixel_fraction', 'hot_pixel_offset', 'hot_pixel_spread'});
[f, draw.flicker] = session(rs, Tb.flicker, {'flicker_rms', 'flicker_smoothness', ...
    'flicker_slow_rms', 'flicker_slow_smoothness', 'flicker_map_rms'});
[t, draw.trend] = session(rs, Tb.trend, {'trend_change'});
[pk, draw.peaks] = session(rs, Tb.peaks, {'burst_rate'});
[g, draw.glitches] = session(rs, Tb.glitches, {'glitches'});
[j, draw.jolts] = session(rs, Tb.jolts, {'jolts_per_hour'});
% A smoothness is a time constant: its factor acts on 1 / (1 - phi), capped.
f.flicker_smoothness = smooth_jitter(Tb.flicker.flicker_smoothness(draw.flicker.index), ...
    draw.flicker.factor(2));
f.flicker_slow_smoothness = smooth_jitter(Tb.flicker.flicker_slow_smoothness(draw.flicker.index), ...
    draw.flicker.factor(4));
% Jolts: a count per movie from the drawn rate per hour (Poisson).
n_jolts = poisson(rs, j.jolts_per_hour * minutes / 60);
draw.minutes = minutes;
% The fast motion (drawn last, so every draw above is as before it existed):
% the shake's std and AR(1) coefficient, its cap, the bends' std, from one of
% S4's 11 clean sessions.
[mo, draw.motion] = session(rs, Tb.motion, {'shake_std', 'shake_smoothness', 'motion_max', 'bend_std'});
mo.shake_smoothness = smooth_jitter(Tb.motion.shake_smoothness(draw.motion.index), draw.motion.factor(2));

camera = { ...
    'camera_supersample', 5, ...                       % v2's camera (sub-pixel averaging)
    'black_level', c.black_level, ...                  % S24: darkest mean pixel 6.2-10.2 grey
    'adc_gain', 1.164, 'adc_gain_prob', 1, ...         % S24: 1.162-1.1645 in all 48 sessions
    'adc_subpixels', 2, ...                            % S4/S24: the simulated px is 2 x 2 sensor px
    'sensor_pattern', true, 'sensor_pattern_subpixels', true, ... % S24: per sensor px 2 x the binned
    'sensor_prnu', c.sensor_prnu, 'sensor_dsnu', c.sensor_dsnu, ...
    'sensor_row_offset', c.sensor_row_offset, 'sensor_col_offset', c.sensor_col_offset, ...
    'hot_pixels', true, ...
    'hot_pixel_fraction', h.hot_pixel_fraction, 'hot_pixel_offset', h.hot_pixel_offset, ...
    'hot_pixel_spread', h.hot_pixel_spread, 'hot_pixel_noise', [0 1], ...
    'vignetting', true, 'vignetting_file', 'vignetting_167-4_norings.mat', ...
    'vignetting_strength', [1.1 1.8], 'vignetting_shift', [0 0.1], 'vignetting_flip', true, ...
    'exposure_target', c.exposure_target, ...
    'flicker', true, 'flicker_rms', f.flicker_rms, 'flicker_smoothness', f.flicker_smoothness, ...
    'flicker_slow_rms', f.flicker_slow_rms, 'flicker_slow_smoothness', f.flicker_slow_smoothness, ...
    'flicker_maps', 3, 'flicker_map_rms', f.flicker_map_rms, 'flicker_map_scale', [30 150], ...
    'brightness_trend', true, 'trend_change', t.trend_change, 'trend_tau', [0.1 2], ...
    'trend_minutes', minutes, ...
    ... % switched off, with the real ranges when switched on
    'dust', false, 'dust_specks', [0 5], 'dust_size', [2 4] .* [1/1.2 1.2], ...
    'dust_contrast', [0.05 0.2] .* [1/1.2 1.2], 'dust_blurry', [0 3], ...
    'dust_blurry_size', [20 120], 'dust_blurry_contrast', [0.02 0.15], ...
    'oof_layer', false, 'oof_halfpoint', [12 48] .* [1/1.2 1.2], ...
    'oof_edge_factor', [1.3 2] .* [1/1.2 1.2], 'oof_gain', [1/1.2 1.2], ...
    'oof_lag', [0 5], 'oof_walk', [0 2]};

scene = { ...
    'smooth_background', true, ...                     % a real scene is not rounded before the camera
    'scene_noise', false, ...                          % the noise is the sensor's (v2)
    'burst_rate', pk.burst_rate, 'burst_fps', fps, ... % S25: 2.5-13.9 per minute
    'burst_glow_source', 'background', ...             % S25: the light is diffuse (somas 0-8 %)
    'burst_neuropil_dff', [0.026 0.305] .* [1/1.2 1.2], ... % S25: p10 2.6 %, p90 30.5 %
    'burst_decay', [2.0 4.6] / log(2) .* [1/1.2 1.2], ...    % S25: half-decay 2.0-4.6 frames
    'burst_duration', [1 5], 'burst_ramp', 1, 'burst_gap', 10, 'burst_interval', [2 4], ...
    'burst_speed', [40 120], 'burst_fraction', [0.1 0.3], 'burst_radius', [60 200], ...
    'burst_amplitude', [0.3 1], 'burst_glow_sigma', 150};

moving = { ...
    'slow_drift', true, 'slow_drift_minutes', minutes, ...
    'slow_drift_rate', [0.03 0.2] .* [1/1.2 1.2], ...        % S4, px/min
    'slow_drift_scale', [0.007 0.08] .* [1/1.2 1.2], ...     % %/min
    'slow_drift_rotation', [0.005 0.02] .* [1/1.2 1.2], ...  % deg/min
    'slow_drift_shear', [0.01 0.047] .* [1/1.2 1.2], ...     % %/min
    'slow_drift_bend_rate', [0.005 0.05 * 1.2], ...          % up to 0.05 px/min
    'slow_drift_wander', [0.2 0.8], 'slow_drift_bend_width', [50 150], ...
    'jolts', n_jolts, 'jolt_size', [0.87 3.47] .* [1/1.2 1.2], ...  % S4: largest event per session
    'jolt_rise', [1 4], 'jolt_return', [2 26] .* [1/1.2 1.2], ...   % S4: events last 0.1-1.3 s
    'jolt_defocus', 0, 'jolt_brightness', [0.02 0.2], ...
    'glitches', g.glitches, 'glitch_run', [1 8 * 1.2]};             % S4: runs of 1-8 frames

% The fast motion at the real range (S4): a stationary shake and broad bends,
% no regimes, no local bends, no jumps (the rare real events are the jolts
% above). The regimes driver offers it as its motion setting 'real' beside
% movie v2's regimes ('v2').
motion = { ...
    'within_session_translation_std', mo.shake_std, ...      % S4: fast 99th pct 0.45-1.69 px / 3.035
    'within_session_motion_smoothness', mo.shake_smoothness, ... % S4: from the step's 99.9th pct
    'within_session_motion_max', mo.motion_max, ...           % S4: largest displacement 1.56-4.6 px
    'within_session_nonrigid_std', mo.bend_std, ...           % S4: non-affine residual 0.12-0.74 px
    'within_session_nonrigid_grid', 3, ...
    'within_session_rigid_frames', [], 'within_session_local_bends', 0, 'within_session_jumps', 0};

R = struct('nv', {[scene, camera, moving, motion, {'sensor_noise', c.sensor_noise}]}, ...
    'scene', {scene}, 'camera', {camera}, 'moving', {moving}, 'motion', {motion}, ...
    'noise', c.sensor_noise, 'draw', draw);
end

function [v, d] = session(rs, T, names)
% One row of table T and a log-uniform factor in [1/1.2, 1.2] per value.
n = numel(T.(names{1}));
i = 1 + floor(rand(rs) * n);
fac = exp((2 * rand(rs, 1, numel(names)) - 1) * log(1.2));
v = struct();
for k = 1:numel(names)
    v.(names{k}) = T.(names{k})(i) * fac(k);
end
s = T.session;
if iscell(s), s = s{i}; else, s = strtrim(s(i, :)); end
d = struct('index', i, 'session', char(s), 'factor', fac);
end

function phi = smooth_jitter(phi0, fac)
% The factor on the time constant 1 / (1 - phi).
tau = 1 / (1 - phi0) * fac;
phi = min(1 - 1 / tau, 0.9999);
end

function n = poisson(rs, lambda)
% Knuth's method (small means).
L = exp(-lambda); n = 0; p = rand(rs);
while p > L
    n = n + 1;
    p = p * rand(rs);
end
end
