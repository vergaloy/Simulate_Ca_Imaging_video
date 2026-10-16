function M = dct_interp_matrix(N, s)
%% Band-limited upsampling of N samples to s*N, as one matrix.
%
% The simulated scene only exists on the pixel grid. To move it by a fraction
% of a pixel it has to be given values between the pixels, and HOW that is done
% decides whether a correlation engine can measure the motion without bias.
% Bilinear (or cubic) interpolation is not band-limited: the copies of the
% spectrum it leaves above the Nyquist frequency fold back when the frame is
% sampled again, by an amount that depends on the fractional shift, and every
% engine is pulled about 0.1 px toward whole pixels. Supersampling and binning
% alone do not cure it.
%
% The cure is a scene with no content above Nyquist. This is the band-limited
% interpolant of the samples mirrored at both ends (half-sample symmetric, so
% the periodic extension has no jump at the border), i.e. the inverse DCT-II
% evaluated between the samples:
%
%   x(t) = X(0)/N + 2/N * sum_{k=1}^{N-1} X(k) cos(pi*k*(2t+1)/(2N))
%   X(k) = sum_n x(n) cos(pi*k*(2n+1)/(2N))
%
% with t the 0-based position in pixels. The Nyquist term of the mirrored
% sequence is exactly zero, so nothing has to be split.
%
% Rows are the s*N points t_j = (j - (s-1)/2)/s, j = 0..s*N-1: the centres of
% an s x s grid of sub-pixels inside each pixel. For odd s the middle sub-pixel
% sits on the pixel centre. For s = 1, M is the identity up to rounding.
%
%   fine = dct_interp_matrix(Ly, s) * img * dct_interp_matrix(Lx, s).'
%
% Author: Pablo Vergara

n = (0:N-1);
t = ((0:s*N-1)' - (s-1)/2) / s;
k = (1:N-1);
Cn = cos(pi * (2*n' + 1) * k / (2*N));      % [N   N-1]
Ct = cos(pi * (2*t  + 1) * k / (2*N));      % [s*N N-1]
M = 1/N + (2/N) * (Ct * Cn.');
end
