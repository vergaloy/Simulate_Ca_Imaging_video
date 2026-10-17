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
% Author: Pablo Vergara

[Ly, Lx, n] = size(FV);
s = cam.s;
out = zeros(Ly, Lx, n, 'uint8');
noisy = nargin > 3 && sensor_noise > 0;
for k = 1:n
    scene = cam.My * double(FV(:, :, k)) * cam.Mx.';
    [fx, fy] = within_session_field(cam.mo, frames(k), cam.xs, cam.ys, cam.B);
    fine = interp2(scene, cam.Xf - s*fx, cam.Yf - s*fy, 'linear', 0);
    binned = squeeze(mean(mean(reshape(fine, s, Ly, s, Lx), 1), 3));
    if noisy
        binned = binned + sensor_noise * randn(rs, Ly, Lx);
    end
    out(:, :, k) = uint8(binned);
end
end
