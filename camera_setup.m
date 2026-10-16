function cam = camera_setup(frame_size, s, mo)
%% Everything apply_camera_motion needs, computed once per session.
%
% The camera model: the scene is the band-limited interpolant of the simulated
% frame (dct_interp_matrix); each pixel of the recorded frame averages that
% scene, moved by the within-session field, at the centres of s x s sub-pixels.
% That is a camera integrating light over its pixels, and it moves the image by
% fractions of a pixel without pulling the motion toward whole pixels.
%
% Inputs:
%   frame_size - [Ly Lx] of the frame
%   s          - sub-pixels per pixel along each axis (odd keeps the middle
%                sub-pixel on the pixel centre; 1 = one point per pixel)
%   mo         - the motion, as within_session_field takes it
%
% Author: Pablo Vergara

Ly = frame_size(1); Lx = frame_size(2);
cam.s = s;
cam.My = dct_interp_matrix(Ly, s);                      % [s*Ly Ly]
cam.Mx = dct_interp_matrix(Lx, s);                      % [s*Lx Lx]
% Sub-pixel centres in pixel coordinates (1-based): sub-pixel J of the fine
% grid sits at (J - (s+1)/2)/s + 1.
cam.xs = ((1:s*Lx) - (s+1)/2) / s + 1;
cam.ys = ((1:s*Ly)' - (s+1)/2) / s + 1;
[cam.Xf, cam.Yf] = meshgrid(1:s*Lx, 1:s*Ly);            % fine-grid indices
cam.mo = mo;
cam.B = [];
if ~isempty(mo.ctrl_xy)
    [X, Y] = meshgrid(cam.xs, cam.ys);
    cam.B = tps_basis_at(mo.ctrl_xy, [X(:), Y(:)]);
end
end
