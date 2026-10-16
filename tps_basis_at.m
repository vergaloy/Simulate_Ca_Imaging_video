function B = tps_basis_at(ctrl_xy, q)
%% Thin-plate-spline basis at arbitrary points.
%
% The same spline as tps_dense_basis, which only evaluates it on the integer
% pixel grid. The camera model (apply_camera_motion) needs it between the
% pixels, at the centres of the sub-pixels it samples.
%
%   ctrl_xy - [n 2] control points, (x, y) in pixels, 1-based
%   q       - [m 2] query points, (x, y), same coordinates
%   B       - [m n]: the field at q is B * c for control-point values c [n 1]
%
% Author: Pablo Vergara

ctrl = double(ctrl_xy);
n = size(ctrl, 1);
d2 = (ctrl(:,1) - ctrl(:,1)').^2 + (ctrl(:,2) - ctrl(:,2)').^2;
P = [ones(n,1), ctrl];
L = [tps_kernel(d2), P; P', zeros(3,3)];
A = L \ [eye(n); zeros(3,n)];                  % [n+3 n]

q = double(q);
B = zeros(size(q,1), n);
for i = 1:n
    Uq = tps_kernel((q(:,1) - ctrl(i,1)).^2 + (q(:,2) - ctrl(i,2)).^2);
    B = B + Uq * A(i,:);
end
B = B + ones(size(q,1),1) * A(n+1,:) + q(:,1) * A(n+2,:) + q(:,2) * A(n+3,:);
end


function out = tps_kernel(d2)
out = zeros(size(d2));
m = d2 > 0;
out(m) = d2(m) .* log(sqrt(d2(m)) + 1e-12);
end
