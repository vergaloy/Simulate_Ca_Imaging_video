function out = apply_camera_motion(FV, frames, cam)
%% Record moving frames through the camera model (see camera_setup).
%
% For each frame: the band-limited scene on the s x s sub-pixel grid, sampled
% at p - f(p) by bilinear interpolation on that fine grid (the scene is smooth
% there, so bilinear costs almost nothing), averaged over each pixel's s x s
% sub-pixels, rounded to uint8. Outside the frame the fill is 0, as in
% apply_nonrigid_motion; leave a margin wider than the largest displacement.
%
% Unlike the two-warp path it replaces, the frame is interpolated ONCE and is
% not rescaled (no v2uint8): a still frame keeps its grey levels, so the movie
% and its motion-free twin (the same call with no motion) share one scale.
%
% Inputs:
%   FV     - [Ly Lx n] uint8 frames of the still scene
%   frames - [1 n] their indices within the session
%   cam    - from camera_setup
%
% Author: Pablo Vergara

[Ly, Lx, n] = size(FV);
s = cam.s;
out = zeros(Ly, Lx, n, 'uint8');
for k = 1:n
    scene = cam.My * double(FV(:, :, k)) * cam.Mx.';
    [fx, fy] = within_session_field(cam.mo, frames(k), cam.xs, cam.ys, cam.B);
    fine = interp2(scene, cam.Xf - s*fx, cam.Yf - s*fy, 'linear', 0);
    out(:, :, k) = uint8(squeeze(mean(mean(reshape(fine, s, Ly, s, Lx), 1), 3)));
end
end
