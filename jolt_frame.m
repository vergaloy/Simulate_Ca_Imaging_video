function fine = jolt_frame(F, t, cam, fx, fy)
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
if J.defocus(t) > 0
    rest = imgaussfilt(rest, J.defocus(t));
end
rest = cam.My * rest * cam.Mx.';
m = J.n_sub(t);
for q = 1:m
    fine = fine + interp2(rest, cam.Xf - s*(fx + J.sub_x(q, t)), ...
        cam.Yf - s*(fy + J.sub_y(q, t)), 'linear', 0) / m;
end
fine = J.gain(t) * fine;
end
