function Yout = apply_nonrigid_motion(Y, B, cdx, cdy)
%% Warp each frame by its own dense displacement field.
% B is the spline basis from tps_dense_basis; cdx/cdy are [n T].
[Ly, Lx, T] = size(Y);
Yout = zeros(size(Y),'like',Y);
for k = 1:T
    D = cat(3, reshape(B*cdx(:,k), Ly, Lx), reshape(B*cdy(:,k), Ly, Lx));
    Yout(:,:,k) = imwarp(Y(:,:,k), single(D), 'linear', 'FillValues', 0);
end
end
