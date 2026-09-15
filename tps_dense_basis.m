function B = tps_dense_basis(ctrl_xy, frame_size, smoothing)
%% Thin-plate-spline basis mapping control-point values to every pixel.
%
% Returns [Ly*Lx  n], so a frame's dense field is one matrix multiply:
%   field = reshape(B * cdx(:,t), Ly, Lx)
%
% The basis depends only on where the control points are, not on how far they
% move, so it is built once and reused for every frame of the session.
%
% A spline rather than an imresize of the control grid, deliberately. NoRMCorre
% upsamples its patch shifts with imresize; simulating the deformation the same
% way would hand the corrector exactly the family of fields it can represent, and
% the test would flatter it.
%
% Author: Pablo Vergara

if nargin<3 || isempty(smoothing), smoothing = 0; end
ctrl = double(ctrl_xy);
n = size(ctrl,1);
Ly = frame_size(1); Lx = frame_size(2);

[QX, QY] = meshgrid(1:Lx, 1:Ly);
q = [QX(:), QY(:)];

K = tps_kernel(pairwise_dist2(ctrl, ctrl)) + smoothing*eye(n);
P = [ones(n,1), ctrl];
L = [K, P; P', zeros(3,3)];

rhs = [eye(n); zeros(3,n)];
A = L \ rhs;
if any(~isfinite(A(:))), A = pinv(L)*rhs; end

Uq = tps_kernel(pairwise_dist2(q, ctrl));
B  = [Uq, ones(size(q,1),1), q] * A;
end


function out = tps_kernel(d2)
out = zeros(size(d2));
m = d2 > 0;
out(m) = d2(m) .* log(sqrt(d2(m)) + 1e-12);
end


function d2 = pairwise_dist2(a, b)
d2 = sum(a.^2,2) + sum(b.^2,2)' - 2*(a*b');
d2(d2<0) = 0;
end
