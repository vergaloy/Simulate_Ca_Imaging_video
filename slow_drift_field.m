function [fx, fy] = slow_drift_field(Dr, t, X, Y)
%% The slow drift (make_slow_drift) at frame t, at points (X, Y).
%
%   D(p, t) = r(t) + G(t) (p - c) + sum_k a_k(t) g_k(p) [cos ang_k, sin ang_k]
%
% X, Y: any equal shapes, pixels, 1-based (x = column, y = row).
%
% Author: Pablo Vergara

c = Dr.centre;
G = Dr.G(:, t);
U = X - c(1); V = Y - c(2);
fx = Dr.r(1, t) + G(1) * U + G(2) * V;
fy = Dr.r(2, t) + G(3) * U + G(4) * V;
for k = 1:size(Dr.bends, 1)
    a = Dr.bend_amp(k, t);
    if a == 0
        continue
    end
    b = Dr.bends(k, :);
    g = exp(-((X - b(1)).^2 + (Y - b(2)).^2) / (2 * b(3)^2));
    fx = fx + (a * cos(b(4))) * g;
    fy = fy + (a * sin(b(4))) * g;
end
end
