function [fx, fy] = within_session_field(mo, t, xs, ys, B)
%% The within-session displacement at frame t, on a grid of points.
%
% Everything that moves the tissue inside a session, composed into one field:
%
%   f(p, t) = shake(t) + jump(t) + w(t) * (bend(p, t) + local bends(p, t))
%
%   shake  the AR(1) translation (make_xy_motion_ar)          mo.dx, mo.dy
%   jump   the sudden steps (make_motion_regimes)             mo.R.jump_xy
%   w      the regime weight, 0 rigid .. 1 bend               mo.R.bend_weight
%   bend   the 3x3 thin-plate spline (make_nonrigid_motion_ar) B * mo.cdx(:,t)
%   local  the Gaussian bumps (make_motion_regimes)           mo.R.bumps
%
% f is the distance the tissue moved, in pixels: the frame shows at p what the
% still tissue had at p - f(p) (same sign as apply_nonrigid_motion).
%
% Inputs:
%   mo     - struct with dx, dy [1 T], ctrl_xy/cdx/cdy (empty if no bend), R
%   t      - frame index within the session
%   xs     - [1 nx] x coordinates (columns), pixels, 1-based
%   ys     - [ny 1] y coordinates (rows)
%   B      - tps_basis_at(mo.ctrl_xy, [X(:) Y(:)]) for [X,Y] = meshgrid(xs,ys),
%            or [] when there is no spline bend
% Outputs:
%   fx, fy - [ny nx]
%
% Author: Pablo Vergara

nx = numel(xs); ny = numel(ys);
fx = (mo.dx(t) + mo.R.jump_xy(1, t)) * ones(ny, nx);
fy = (mo.dy(t) + mo.R.jump_xy(2, t)) * ones(ny, nx);
w = mo.R.bend_weight(t);
if w == 0
    return
end

bx = zeros(ny, nx); by = zeros(ny, nx);
if ~isempty(B) && ~isempty(mo.cdx)
    bx = reshape(B * mo.cdx(:, t), ny, nx);
    by = reshape(B * mo.cdy(:, t), ny, nx);
end

bumps = mo.R.bumps;
for k = find(bumps(:, 7) <= t & t <= bumps(:, 8))'
    b = num2cell(bumps(k, :));
    [cx, cy, sigma, amp, angle, kind, t_on, t_off, sgn] = b{1:9};
    a = amp * sin(pi * (t - t_on + 1) / (t_off - t_on + 2))^2;
    gx = exp(-(xs - cx).^2 / (2*sigma^2));          % [1 nx]
    gy = exp(-(ys - cy).^2 / (2*sigma^2));          % [ny 1]
    if kind == 1
        g = gy * gx;
        bx = bx + (a * cos(angle)) * g;
        by = by + (a * sin(angle)) * g;
    else
        c = sgn * a * sqrt(exp(1)) / sigma;
        bx = bx + c * (gy * (gx .* (xs - cx)));
        by = by + c * ((gy .* (ys - cy)) * gx);
    end
end
fx = fx + w * bx;
fy = fy + w * by;
end
