function [fx, fy] = compose_slow_drift(Dr, t, X, Y, fx, fy)
%% The within-session field (fx, fy) at points (X, Y), composed with the slow drift.
%
% The tissue drifts slowly (make_slow_drift), and the fast motion (shake,
% bends, jumps) moves the drifted tissue: warp by the drift first, then by the
% fast field f. The composite, at p:
%
%   f(p) + D(p - f(p))
%
% which is patchforge's poc.fields.compose(drift, fast). Composed, not added.
%
% Author: Pablo Vergara

[gx, gy] = slow_drift_field(Dr, t, X - fx, Y - fy);
fx = fx + gx;
fy = fy + gy;
end
