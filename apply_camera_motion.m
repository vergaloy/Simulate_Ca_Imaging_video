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
% Author: Pablo Vergara

[Ly, Lx, n] = size(FV);
s = cam.s;
out = zeros(Ly, Lx, n, 'uint8');
noisy = nargin > 3 && sensor_noise > 0;
has_imp = isfield(cam, 'imp') && ~isempty(cam.imp);
for k = 1:n
    [fx, fy] = within_session_field(cam.mo, frames(k), cam.xs, cam.ys, cam.B);
    if has_imp && cam.imp.I.index(frames(k)) > 0
        fine = impact_frame(double(FV(:, :, k)), frames(k), cam, fx, fy);
    else
        scene = cam.My * double(FV(:, :, k)) * cam.Mx.';
        fine = interp2(scene, cam.Xf - s*fx, cam.Yf - s*fy, 'linear', 0);
    end
    binned = squeeze(mean(mean(reshape(fine, s, Ly, s, Lx), 1), 3));
    if noisy
        binned = binned + sensor_noise * randn(rs, Ly, Lx);
    end
    out(:, :, k) = uint8(binned);
end
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
