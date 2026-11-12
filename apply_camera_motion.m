function out = apply_camera_motion(FV, frames, cam, sensor_noise, rs)
%% Record moving frames through the camera model (see camera_setup).
%
% For each frame: the band-limited scene on the s x s sub-pixel grid, sampled
% at p - f(p) by bilinear interpolation on that fine grid (the scene is smooth
% there, so bilinear costs almost nothing), averaged over each pixel's s x s
% sub-pixels, plus the sensor noise if any, rounded to uint8. Outside the frame
% the fill is 0, as in apply_nonrigid_motion; leave a margin wider than the
% largest displacement.
%
% The noise goes in before the rounding, as in a camera, where the read-out
% is digitised once. Rounded first, a smooth noise-free scene is a staircase
% of flat steps whose edges can only sit between pixels, so they move in
% whole-pixel steps and pull a correlation engine toward whole pixels (about
% 0.08 px on the regimes movie); noise added after that does not undo it.
%
% Unlike the two-warp path it replaces, the frame is interpolated ONCE and is
% not rescaled (no v2uint8): a still frame keeps its grey levels, so the movie
% and its motion-free twin (the same call with no motion) share one scale.
%
% Inputs:
%   FV     - [Ly Lx n] uint8 frames of the still scene
%   frames - [1 n] their indices within the session
%   cam    - from camera_setup
%   sensor_noise - optional: std of white Gaussian noise, grey levels, one
%            draw per frame (randn(rs, Ly, Lx)) in frame order. 0 = none.
%   rs     - the RandStream it is drawn from
%
% Frames inside an impact (cam.imp, set by Simulate_Ca_video when impacts are
% on) are built by impact_frame below; every other frame as above.
%
% Slow drift (cam.drift, make_slow_drift) is composed with the field of every
% frame (compose_slow_drift); frames inside a jolt (cam.jolt, make_jolts) are
% built by jolt_frame.
%
% With cam.op (make_camera_optics, set by Simulate_Ca_video when any optics
% option is on), what the camera adds that does not move with the tissue:
% an out-of-focus layer that moves on its own (camera_layer_frame; not inside
% impacts), then vignetting, dust, brightness and the sensor's fixed pattern
% and hot pixels (camera_optics_frame), and, with the sensor noise, the hot
% pixels' own noise (after the sensor noise, from cam.op.rs_hot). With
% cam.op.oof.save_file set, every frame's out-of-focus light is written there
% (/L, [Ly Lx T] double), for the replay proof. Its black level is part of the
% sensor's offset; with cam.op.adc_gain g ~= 1 the read-out is digitised in
% steps of g and multiplied by g (missing codes, make_camera_optics) before the
% 8-bit rounding, in every frame, noisy or not; with cam.op.adc_subpixels n > 1
% and the sensor noise on, each pixel is read by n x n sensor pixels, each with
% noise n * sensor_noise (so their mean has sensor_noise), its own ADC and its
% own 8-bit code, and the codes averaged (n^2 noise draws per frame from rs;
% the hot pixels' noise is added before). A noise-free frame reads the same
% through one ADC or n^2.
%
% Author: Pablo Vergara

[Ly, Lx, n] = size(FV);
s = cam.s;
out = zeros(Ly, Lx, n, 'uint8');
noisy = nargin > 3 && sensor_noise > 0;
has_imp = isfield(cam, 'imp') && ~isempty(cam.imp);
has_drift = isfield(cam, 'drift') && ~isempty(cam.drift);
has_jolt = isfield(cam, 'jolt') && ~isempty(cam.jolt);
has_op = isfield(cam, 'op') && ~isempty(cam.op);
has_oof = has_op && ~isempty(cam.op.oof);
has_adc = has_op && isfield(cam.op, 'adc_gain') && cam.op.adc_gain ~= 1;
save_oof = has_oof && ~isempty(cam.op.oof.save_file);
for k = 1:n
    [fx, fy] = within_session_field(cam.mo, frames(k), cam.xs, cam.ys, cam.B);
    if has_drift
        [fx, fy] = compose_slow_drift(cam.drift, frames(k), cam.Xp, cam.Yp, fx, fy);
    end
    if has_imp && cam.imp.I.index(frames(k)) > 0
        fine = impact_frame(double(FV(:, :, k)), frames(k), cam, fx, fy);
    elseif has_jolt && cam.jolt.J.index(frames(k)) > 0
        [fine, L] = jolt_frame(double(FV(:, :, k)), frames(k), cam, fx, fy);
        if save_oof
            h5write(cam.op.oof.save_file, '/L', L, [1 1 frames(k)], [Ly Lx 1]);
        end
    else
        scene = cam.My * double(FV(:, :, k)) * cam.Mx.';
        if has_oof
            [fine, L] = camera_layer_frame(scene, double(FV(:, :, k)), frames(k), cam, fx, fy);
            if save_oof
                h5write(cam.op.oof.save_file, '/L', L, [1 1 frames(k)], [Ly Lx 1]);
            end
        else
            fine = interp2(scene, cam.Xf - s*fx, cam.Yf - s*fy, 'linear', 0);
        end
    end
    binned = squeeze(mean(mean(reshape(fine, s, Ly, s, Lx), 1), 3));
    if has_op
        binned = camera_optics_frame(binned, frames(k), cam.op);
    end
    if noisy && has_adc && cam.op.adc_subpixels > 1
        % Read out by n x n sensor pixels (cam.op.adc_subpixels), each with
        % noise n * sensor_noise and its own ADC, written as an 8-bit code
        % (the rounding that skips the missing codes), then averaged: the
        % simulated pixel is a binned one, as the real recordings binned 2 x 2.
        if ~isempty(cam.op.hot_idx)
            h = cam.op.hot_idx;
            binned(h) = binned(h) + cam.op.hot_std .* randn(cam.op.rs_hot, numel(h), 1);
        end
        nq = cam.op.adc_subpixels;
        acc = zeros(Ly, Lx);
        has_sub = isfield(cam.op, 'dsnu_sub') && ~isempty(cam.op.dsnu_sub);
        for q = 1:nq^2
            v = binned + nq * sensor_noise * randn(rs, Ly, Lx);
            if has_sub, v = v + cam.op.dsnu_sub(:, :, q); end   % S28: per sensor pixel
            acc = acc + double(uint8(adc_read(v, cam.op)));
        end
        binned = acc / nq^2;
    else
        if noisy
            binned = binned + sensor_noise * randn(rs, Ly, Lx);
            if has_op && ~isempty(cam.op.hot_idx)
                h = cam.op.hot_idx;
                binned(h) = binned(h) + cam.op.hot_std .* randn(cam.op.rs_hot, numel(h), 1);
            end
        end
        if has_adc
            binned = adc_read(binned, cam.op);
        end
    end
    out(:, :, k) = uint8(binned);
end
end

function y = adc_read(x, op)
%% The ADC's digital gain (make_camera_optics): digitised in steps of g, times g.
g = op.adc_gain; c = op.adc_offset;
y = g * round((x - c) / g) + c;
end

function fine = impact_frame(F, t, cam, fx, fy)
%% One frame inside an impact (make_impacts), on the camera's fine grid.
%
% The frame is split into a part fixed to the optics, the baseline's low-pass
% part plus the frame's mean brightness (cam.imp.LB, in the frame's grey
% scale), and the moving rest: the baseline's high-pass part, the neurons and
% the background sources, zero-mean. At each sub-frame instant the rest is
% moved by the tissue's field plus the impact's slide and weighted by w, and
% the foreign structure (cam.imp.Hf_s, already on the fine grid) is moved by
% its own offset and weighted by 1 - w; the instants are averaged (motion
% blur). Outside the frame both moving parts are 0, so only the fixed part
% shows there.
s = cam.s; I = cam.imp.I;
fixed = cam.imp.LB + (mean(F, 'all') - mean(cam.imp.LB, 'all'));
fine = cam.My * fixed * cam.Mx.';
nb = size(I.sub_w, 1);
w = I.sub_w(:, t);
% Real scene: one interpolation per distinct position, its weights summed.
if any(w > 0)
    rest = cam.My * (F - fixed) * cam.Mx.';
    [pos, ~, g] = unique([I.sub_x(:, t), I.sub_y(:, t)], 'rows', 'stable');
    for q = 1:size(pos, 1)
        wq = sum(w(g == q)) / nb;
        if wq > 0
            fine = fine + wq * interp2(rest, cam.Xf - s*(fx + pos(q, 1)), ...
                cam.Yf - s*(fy + pos(q, 2)), 'linear', 0);
        end
    end
end
% Foreign structure.
if any(w < 1)
    [pos, ~, g] = unique([I.sub_fx(:, t), I.sub_fy(:, t)], 'rows', 'stable');
    for q = 1:size(pos, 1)
        wq = sum(1 - w(g == q)) / nb;
        if wq > 0
            fine = fine + wq * interp2(cam.imp.Hf_s, cam.Xf - s*pos(q, 1), ...
                cam.Yf - s*pos(q, 2), 'linear', 0);
        end
    end
end
end
