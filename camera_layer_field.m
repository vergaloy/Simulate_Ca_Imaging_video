function [gx, gy] = camera_layer_field(cam, t, fx, fy)
%% The out-of-focus layer's field at frame t on the camera's fine grid.
%
%   g(p, t) = gain * f(p, t - lag) + walk(t)
%
% f is the tissue's field (fx, fy at frame t itself); at a fractional frame it
% is the linear blend of the frames on either side, and before the first frame
% it is the first frame's. cam.op.oof from make_camera_optics.
%
% Author: Pablo Vergara

o = cam.op.oof;
if o.lag == 0
    hx = fx; hy = fy;
else
    tl = t - o.lag;
    k0 = floor(tl); a = tl - k0;
    if k0 < 1
        k0 = 1; a = 0;
    end
    [hx, hy] = at(cam, k0, t, fx, fy);
    if a > 0
        [bx, by] = at(cam, k0 + 1, t, fx, fy);
        hx = (1 - a) * hx + a * bx;
        hy = (1 - a) * hy + a * by;
    end
end
gx = o.gain * hx + o.walk(1, t);
gy = o.gain * hy + o.walk(2, t);
end

function [ux, uy] = at(cam, k, t, fx, fy)
% The tissue's field at frame k, as apply_camera_motion computes it at frame t.
if k == t
    ux = fx; uy = fy;
else
    [ux, uy] = within_session_field(cam.mo, k, cam.xs, cam.ys, cam.B);
end
end
