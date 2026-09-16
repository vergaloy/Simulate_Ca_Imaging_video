function Yout = apply_nonrigid_motion(Y, B, cdx, cdy)
%% Warp each frame by its own dense displacement field.
% B is the spline basis from tps_dense_basis; cdx/cdy are [n T].
%
% THE SIGN. imwarp treats its displacement field as a BACKWARD map: with
% D(:,:,1) = +5 the content of the image moves five pixels the OTHER way. So the
% field is negated here, and cdx/cdy then mean what they appear to mean -- the
% distance the tissue actually moves. Without this the recorded ground truth is
% the negative of the motion applied, and anything measured against it comes out
% anti-correlated for no visible reason.
[Ly, Lx, T] = size(Y);
Yout = zeros(size(Y),'like',Y);
for k = 1:T
    D = -cat(3, reshape(B*cdx(:,k), Ly, Lx), reshape(B*cdy(:,k), Ly, Lx));
    Yout(:,:,k) = imwarp(Y(:,:,k), single(D), 'linear', 'FillValues', 0);
end
end
