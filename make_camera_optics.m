function op = make_camera_optics(frame_size, T, p, seed)
%% What the camera adds that does not move with the tissue (all optional).
%
% Drawn once per session; apply_camera_motion applies it to every frame after
% the motion. Three groups, each off unless switched on:
%
% Sensor (path M3), fixed to the camera:
%   hot pixels      isolated pixels with an additive offset (heavy-tailed,
%                   lognormal), still carrying the signal and the noise, plus a
%                   noise of their own (dark-current shot noise, std
%                   sqrt(c * offset)) when the sensor noise is on
%   fixed pattern   per-pixel gain texture (PRNU), per-pixel offset texture
%                   (DSNU), row offsets and column offsets
%   dust            dark specks on the sensor or filter (sharp) and faint broad
%                   dimming from dust on a lens (a soft disk), multiplicative
%                   and static
% Optics (path M4), fixed to the camera:
%   vignetting      a map extracted from a real recording with strong
%                   vignetting (vignetting_file; see its own 'source'),
%                   multiplicative, optionally flipped, shifted and made
%                   stronger or weaker (V .^ strength)
%   out-of-focus    the scene split into in-focus structure and out-of-focus
%   layer           light by a low-pass filter drawn per session (Gaussian,
%                   difference of Gaussians, morphological opening; half-point
%                   drawn log-uniformly), whose blur can grow toward the frame
%                   edge; the light moves with its own field,
%                       g(p, t) = gain * f(p, t - lag) + walk(t),
%                   f the tissue's field, walk an AR(1) path of its own:
%                   correlated with the tissue's motion but distinct
% Brightness (path M6), multiplicative on the light:
%   flicker         uniform, an AR(1) series, plus smooth maps varying over the
%                   frame, each with an AR(1) series of its own
%   trend           a slow bleaching-like change, exponential in time, from 1
%                   to 1 + change at the last frame
%
% A frame is recorded as (apply_camera_motion, camera_optics_frame)
%
%   light  = binned .* V .* D .* gain(t)          gain(t) = trend(t) *
%   sensor = light .* (1 + prnu) + offset                   (1 + global(t)) *
%   frame  = uint8(sensor + noise)                          (1 + sum_k c_k(t) M_k)
%
% offset = row + column + dsnu + hot + black level. Every group draws from a
% stream of its own (seed + 8100 sensor, 8200 dust, 8300 vignetting, 8400
% out-of-focus layer, 8500 brightness; 8600 the hot pixels' noise, per frame;
% 8700 black level, 8800 the ADC), so switching one on changes nothing else,
% and a run without them is bit for bit the same.
%
% Read-out (S24), fixed to the camera:
%   black level     the sensor's pedestal, grey levels added to the offset:
%                   with no light a pixel reads it, still noisy, so dark
%                   corners do not clip at 0
%   ADC gain        the camera digitises in steps of g grey (integer codes k)
%                   and multiplies by a digital gain g > 1 before the 8-bit
%                   frame: frame = round(g k + c), k = round((sensor + noise -
%                   c) / g), so one 8-bit code in every g / (g - 1) is never
%                   used (missing codes; the real recordings skip one in 7).
%                   Where the noise is under a few grey, the expected frame is
%                   the scene plus a sawtooth of period g / (g - 1) grey of the
%                   scene's own grey level: a smooth background (the
%                   vignetting) shows its iso-grey contours as terraces, which
%                   background removal turns into concentric rings fixed to the
%                   camera. The noise dithers them away (amplitude times
%                   exp(-2 pi^2 s^2 (g - 1)^2 / g^2), s the noise std at the
%                   ADC). With adc_subpixels n > 1 each simulated pixel is
%                   read by n x n sensor pixels (noise n times the pixel's,
%                   each digitised, then averaged), as a recording binned
%                   n x n: the terraces are then damped by the sensor
%                   pixel's noise, as in the real chain.
%
% Inputs:
%   frame_size - [d1 d2], the saved frame the camera records
%   T          - frames in the session
%   p          - the options (Simulate_Ca_video's optics_options): see there
%   seed       - opt.seed + session
% Output, struct op (empty fields for what is off):
%   V, D, prnu, dsnu [d1 d2]; row [d1 1]; col [1 d2]; hot_idx, hot_offset,
%   hot_std [n 1]; VD = V .* D; P = 1 + prnu; O = offset map;
%   gain_global, gain_trend [1 T]; gain_maps [d1 d2 K]; gain_coef [K T];
%   oof (struct or []); params (every drawn value); rs_hot (RandStream);
%   black_level (grey, 0 = none, also in O); adc_gain (1 = none), adc_offset,
%   adc_subpixels.
%
% Author: Pablo Vergara

d1 = frame_size(1); d2 = frame_size(2);
inset = max(0, round(p.fov_inset));
vis_r = inset + 1:d1 - inset; vis_c = inset + 1:d2 - inset;   % what is kept
op = struct('V', [], 'D', [], 'prnu', [], 'dsnu', [], 'row', [], 'col', [], ...
    'hot_idx', zeros(0, 1), 'hot_offset', zeros(0, 1), 'hot_std', zeros(0, 1), ...
    'gain_global', [], 'gain_trend', [], 'gain_maps', [], 'gain_coef', [], ...
    'oof', [], 'params', struct(), 'VD', 1, 'P', 1, 'O', 0, 'rs_hot', [], ...
    'black_level', 0, 'adc_gain', 1, 'adc_offset', 0, 'adc_subpixels', 1);

%% Sensor: hot pixels and the fixed pattern (seed + 8100)
if p.hot_pixels || p.sensor_pattern
    rs = RandStream('mt19937ar', 'Seed', seed + 8100);
    O = zeros(d1, d2);
    if p.hot_pixels
        frac = log_uniform(rs, p.hot_pixel_fraction);
        n = round(frac * d1 * d2);
        idx = sort(randperm(rs, d1 * d2, n))';
        med = uniform(rs, p.hot_pixel_offset);
        spread = uniform(rs, p.hot_pixel_spread);
        off = med * exp(spread * randn(rs, n, 1));
        c = uniform(rs, p.hot_pixel_noise);
        op.hot_idx = idx; op.hot_offset = off; op.hot_std = sqrt(c * off);
        O(idx) = O(idx) + off;
        op.params.hot_fraction = frac; op.params.hot_n = n;
        op.params.hot_offset_median = med; op.params.hot_offset_spread = spread;
        op.params.hot_noise_c = c;
    end
    if p.sensor_pattern
        s_prnu = uniform(rs, p.sensor_prnu);
        s_dsnu = uniform(rs, p.sensor_dsnu);
        s_row = uniform(rs, p.sensor_row_offset);
        s_col = uniform(rs, p.sensor_col_offset);
        op.prnu = s_prnu * randn(rs, d1, d2);
        op.dsnu = s_dsnu * randn(rs, d1, d2);
        op.row = s_row * randn(rs, d1, 1);
        op.col = s_col * randn(rs, 1, d2);
        O = O + op.dsnu + op.row + op.col;
        op.P = 1 + op.prnu;
        op.params.prnu_std = s_prnu; op.params.dsnu_std = s_dsnu;
        op.params.row_std = s_row; op.params.col_std = s_col;
    end
    op.O = O;
end

%% Dust (seed + 8200)
if p.dust
    rs = RandStream('mt19937ar', 'Seed', seed + 8200);
    [X, Y] = meshgrid(1:d2, 1:d1);
    D = ones(d1, d2);
    n = randi_range(rs, p.dust_specks);
    sp = zeros(n, 6);                            % cx cy fwhm aspect angle contrast
    for k = 1:n
        cx = vis_c(1) + rand(rs) * (numel(vis_c) - 1);
        cy = vis_r(1) + rand(rs) * (numel(vis_r) - 1);
        fw = log_uniform(rs, p.dust_size);
        asp = 1 + rand(rs);
        ang = pi * rand(rs);
        con = uniform(rs, p.dust_contrast);
        sp(k, :) = [cx cy fw asp ang con];
        s1 = fw / 2.3548; s2 = s1 * asp;
        u = (X - cx) * cos(ang) + (Y - cy) * sin(ang);
        v = -(X - cx) * sin(ang) + (Y - cy) * cos(ang);
        D = D .* (1 - con * exp(-u.^2 / (2 * s2^2) - v.^2 / (2 * s1^2)));
    end
    m = randi_range(rs, p.dust_blurry);
    bl = zeros(m, 4);                            % cx cy diameter contrast
    for k = 1:m
        cx = vis_c(1) + rand(rs) * (numel(vis_c) - 1);
        cy = vis_r(1) + rand(rs) * (numel(vis_r) - 1);
        dia = log_uniform(rs, p.dust_blurry_size);
        con = uniform(rs, p.dust_blurry_contrast);
        bl(k, :) = [cx cy dia con];
        r = hypot(X - cx, Y - cy);
        edge = max(1, 0.15 * dia);
        D = D .* (1 - con ./ (1 + exp((r - dia / 2) / (edge / 4))));
    end
    op.D = D;
    op.params.dust_specks = sp; op.params.dust_blurry = bl;
end

%% Vignetting (seed + 8300)
if p.vignetting
    rs = RandStream('mt19937ar', 'Seed', seed + 8300);
    S = load(p.vignetting_file, 'V');
    V0 = double(S.V);
    strength = uniform(rs, p.vignetting_strength);
    flip_ud = p.vignetting_flip && rand(rs) < 0.5;
    flip_lr = p.vignetting_flip && rand(rs) < 0.5;
    shift_frac = uniform(rs, p.vignetting_shift);
    shift_ang = 2 * pi * rand(rs);
    if flip_ud, V0 = flipud(V0); end
    if flip_lr, V0 = fliplr(V0); end
    % Scaled to cover what is kept (the field of view), the same scale on
    % both axes; the excess is cropped evenly, moved by the shift.
    h = numel(vis_r); w = numel(vis_c);
    sc = max(h / size(V0, 1), w / size(V0, 2));
    Vr = imresize(V0, sc, 'bilinear');
    Vr = max(Vr, 0);
    sh = shift_frac * w * [cos(shift_ang), sin(shift_ang)];  % px (x, y)
    pad = ceil(max(abs(sh))) + 1;
    Vr = padarray(Vr, [pad pad], 'replicate');
    r0 = floor((size(Vr, 1) - h) / 2) + 1 - round(sh(2));
    c0 = floor((size(Vr, 2) - w) / 2) + 1 - round(sh(1));
    Vk = Vr(r0:r0 + h - 1, c0:c0 + w - 1);
    V = padarray(Vk, [inset inset], 'replicate');
    V = V(1:d1, 1:d2) .^ strength;
    op.V = V;
    op.params.vignetting_strength = strength;
    op.params.vignetting_flip = [flip_ud flip_lr];
    op.params.vignetting_shift_px = round(sh);
    op.params.vignetting_scale = sc;
    op.params.vignetting_file = p.vignetting_file;
end
if ~isempty(op.V) || ~isempty(op.D)
    op.VD = 1;
    if ~isempty(op.V), op.VD = op.VD .* op.V; end
    if ~isempty(op.D), op.VD = op.VD .* op.D; end
end

%% Out-of-focus layer (seed + 8400)
if p.oof_layer
    rs = RandStream('mt19937ar', 'Seed', seed + 8400);
    shapes = cellstr(p.oof_shape);
    o = struct();
    o.shape = shapes{randi(rs, numel(shapes))};
    o.halfpoint = log_uniform(rs, p.oof_halfpoint);
    o.edge_factor = uniform(rs, p.oof_edge_factor);
    o.gain = uniform(rs, p.oof_gain);
    o.lag = uniform(rs, p.oof_lag);
    walk_std = uniform(rs, p.oof_walk);
    phi = p.oof_walk_smoothness;
    e = sqrt(1 - phi^2) * walk_std * randn(rs, 2, T);
    walk = filter(1, [1 -phi], e, [], 2);
    walk = walk - mean(walk, 2);
    if walk_std > 0 && ~isempty(p.oof_walk_max)
        walk = walk / max(1, max(abs(walk), [], 'all') / p.oof_walk_max);
    end
    o.walk_std = walk_std;
    o.walk = walk;                               % [2 T] (x; y), px
    % Blur weight over the frame: 0 at the centre of the field of view, 1 at
    % its corners, growing with the squared radius.
    [X, Y] = meshgrid(1:d2, 1:d1);
    cx = (vis_c(1) + vis_c(end)) / 2; cy = (vis_r(1) + vis_r(end)) / 2;
    rc = hypot(vis_c(end) - cx, vis_r(end) - cy);
    o.edge_weight = min(1, (hypot(X - cx, Y - cy) / rc).^2);
    o.save_file = p.oof_layer_file;
    op.oof = o;
    op.params.oof = rmfield(o, {'walk', 'edge_weight', 'save_file'});
end

%% Brightness (seed + 8500)
if p.flicker || p.brightness_trend
    rs = RandStream('mt19937ar', 'Seed', seed + 8500);
    op.gain_global = zeros(1, T); op.gain_trend = ones(1, T);
    if p.flicker
        rms_g = uniform(rs, p.flicker_rms);
        phi_g = uniform(rs, p.flicker_smoothness);
        op.gain_global = rms_g * unit_ar1(rs, T, phi_g);
        K = round(p.flicker_maps);
        rms_m = uniform(rs, p.flicker_map_rms);
        op.gain_maps = zeros(d1, d2, K); op.gain_coef = zeros(K, T);
        scales = zeros(1, K); phis = zeros(1, K);
        for k = 1:K
            scales(k) = log_uniform(rs, p.flicker_map_scale);
            phis(k) = uniform(rs, p.flicker_smoothness);
            M = imgaussfilt(randn(rs, d1, d2), scales(k), 'Padding', 'symmetric');
            Mv = M(vis_r, vis_c);
            M = (M - mean(Mv, 'all')) / std(Mv, 1, 'all');
            op.gain_maps(:, :, k) = M;
            op.gain_coef(k, :) = rms_m / sqrt(max(K, 1)) * unit_ar1(rs, T, phis(k));
        end
        op.params.flicker_rms = rms_g; op.params.flicker_smoothness = phi_g;
        op.params.flicker_map_rms = rms_m; op.params.flicker_map_scale = scales;
        op.params.flicker_map_smoothness = phis;
    end
    if p.brightness_trend
        change = uniform(rs, p.trend_change);
        tau = uniform(rs, p.trend_tau) * T;
        t = 0:T - 1;
        op.gain_trend = 1 + change * (1 - exp(-t / tau)) / (1 - exp(-(T - 1) / tau));
        op.params.trend_change = change; op.params.trend_tau_frames = tau;
    end
end

%% The hot pixels' own noise, one draw per frame (seed + 8600)
op.rs_hot = RandStream('mt19937ar', 'Seed', seed + 8600);

%% Black level (seed + 8700): a pedestal on the offset
if any(p.black_level(:) > 0)
    rs = RandStream('mt19937ar', 'Seed', seed + 8700);
    op.black_level = uniform(rs, p.black_level);
    op.O = op.O + op.black_level;
    op.params.black_level = op.black_level;
end

%% The ADC's digital gain (seed + 8800): missing codes, terraces
if any(p.adc_gain(:) ~= 1)
    rs = RandStream('mt19937ar', 'Seed', seed + 8800);
    has = rand(rs) < p.adc_gain_prob;
    g = uniform(rs, p.adc_gain);
    c = g * rand(rs);
    if has && g ~= 1
        op.adc_gain = g; op.adc_offset = c;
        op.adc_subpixels = max(1, round(p.adc_subpixels));
    end
    op.params.adc_gain = op.adc_gain; op.params.adc_offset = op.adc_offset;
    op.params.adc_subpixels = op.adc_subpixels;
end
end

function x = uniform(rs, r)
r = r(:)';
if numel(r) == 1 || r(1) == r(end)
    x = r(1);
else
    x = r(1) + rand(rs) * (r(end) - r(1));
end
end

function x = log_uniform(rs, r)
r = r(:)';
if numel(r) == 1 || r(1) == r(end)
    x = r(1);
else
    x = exp(log(r(1)) + rand(rs) * (log(r(end)) - log(r(1))));
end
end

function n = randi_range(rs, r)
r = round(r(:)');
if numel(r) == 1 || r(1) == r(end)
    n = r(1);
else
    n = r(1) + floor(rand(rs) * (r(end) - r(1) + 1));
end
end

function a = unit_ar1(rs, T, phi)
% An AR(1) series of exactly zero mean and unit std over its T frames.
e = randn(rs, 1, T);
a = filter(1, [1 -phi], e);
a = a - mean(a);
s = std(a, 1);
if s > 0, a = a / s; end
end
