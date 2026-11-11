function [fine, L] = camera_layer_frame(scene, F, t, cam, fx, fy)
%% One frame with an out-of-focus layer that moves on its own (make_camera_optics).
%
% The still frame F is split into out-of-focus light L, a low-pass of F
% (camera_layer_lowpass), and in-focus structure F - L. On the camera's fine
% grid the structure moves with the tissue's field f and the light with the
% layer's own field
%
%   g(p, t) = gain * f(p, t - lag) + walk(t)
%
% (f at a fractional frame is the linear blend of the two frames around it;
% before the first frame it is the first frame's). With gain 1, lag 0 and no
% walk this is the frame without the layer, up to rounding.
%
% Inputs:
%   scene  - cam.My * F * cam.Mx.', the band-limited scene on the fine grid
%   F      - [d1 d2] the still frame, double
%   t      - frame within the session
%   cam    - from camera_setup, with cam.op from make_camera_optics
%   fx, fy - the tissue's field at frame t on the fine grid
% Outputs:
%   fine   - the frame on the fine grid, before the pixels average it
%   L      - [d1 d2] the out-of-focus light at frame t (the layer itself)
%
% Author: Pablo Vergara

s = cam.s; o = cam.op.oof;
L = camera_layer_lowpass(F, o);
Ls = cam.My * L * cam.Mx.';
[gx, gy] = camera_layer_field(cam, t, fx, fy);
fine = interp2(scene - Ls, cam.Xf - s*fx, cam.Yf - s*fy, 'linear', 0) + ...
    interp2(Ls, cam.Xf - s*gx, cam.Yf - s*gy, 'linear', 0);
end
