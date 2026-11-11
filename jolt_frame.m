function [fine, L] = jolt_frame(F, t, cam, fx, fy)
%% One frame inside a jolt (make_jolts), on the camera's fine grid.
%
% The frame is split into a part fixed to the optics, the baseline's low-pass
% part plus the frame's mean brightness (cam.jolt.LB, in the frame's grey
% scale; empty = no split, the whole frame moves), and the rest. The fixed
% part moves with the tissue's own field (fx, fy) only, as in every other
% frame; the rest also moves by the jolt's offset, at each of the frame's
% sub-frame instants, and the instants are averaged (motion blur). The rest is
% blurred by the jolt's defocus first, and the whole frame is scaled by its
% gain. Outside the frame the fill is 0, as for every frame.
%
% With the optics' out-of-focus layer (cam.op.oof, make_camera_optics), the
% rest is split as camera_layer_frame splits a frame: the layer's light L
% moves with the layer's own field plus the jolt's offset, the in-focus
% structure with the tissue's field plus the offset (the jolt moves the scope
% against the whole brain). L is returned (empty without the layer).
%
% Inputs:
%   F       - [Ly Lx] double, the still frame
%   t       - its index within the session
%   cam     - from camera_setup, with cam.jolt = struct('J', make_jolts
%             output, 'LB', low-pass part or [])
%   fx, fy  - the tissue's field on the fine grid (apply_camera_motion)
%
% Author: Pablo Vergara

s = cam.s; J = cam.jolt.J;
if isempty(cam.jolt.LB)
    fixed = zeros(size(F));
    fine = zeros(size(cam.Xf));
else
    fixed = cam.jolt.LB + (mean(F, 'all') - mean(cam.jolt.LB, 'all'));
    fine = interp2(cam.My * fixed * cam.Mx.', cam.Xf - s*fx, cam.Yf - s*fy, 'linear', 0);
end
rest = F - fixed;
L = [];
has_oof = isfield(cam, 'op') && ~isempty(cam.op) && ~isempty(cam.op.oof);
if has_oof
    L = camera_layer_lowpass(rest, cam.op.oof);
    rest = rest - L;
    Ls = cam.My * L * cam.Mx.';
    [gx, gy] = camera_layer_field(cam, t, fx, fy);
end
if J.defocus(t) > 0
    rest = imgaussfilt(rest, J.defocus(t));
end
rest = cam.My * rest * cam.Mx.';
m = J.n_sub(t);
for q = 1:m
    jx = J.sub_x(q, t); jy = J.sub_y(q, t);
    fine = fine + interp2(rest, cam.Xf - s*(fx + jx), cam.Yf - s*(fy + jy), 'linear', 0) / m;
    if has_oof
        fine = fine + interp2(Ls, cam.Xf - s*(gx + jx), cam.Yf - s*(gy + jy), 'linear', 0) / m;
    end
end
fine = J.gain(t) * fine;
end
