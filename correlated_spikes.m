function S = correlated_spikes(X, Y, P, F, corr_strength, corr_length_gSig, gSig_fullres, burst_tau)
% CORRELATED_SPIKES Distance-dependent correlated Bernoulli spike trains.
%
%   S = correlated_spikes(X, Y, P, F, corr_strength, corr_length_gSig, gSig_fullres)
%
%   Drop-in replacement for the independent Bernoulli draw performed inside
%   create_random_spike.m. Neurons that sit close to each other fire together
%   more often than chance, with the pairwise correlation decaying
%   exponentially with the distance between their centres:
%
%       rho(d) = corr_strength * exp( -d / (corr_length * gSig) )
%
%   INPUTS
%     X, Y              Neuron centre coordinates, in FULL-RESOLUTION pixels.
%                       Vectors of length N (any orientation).
%     P                 Per-neuron spike probability per frame, length N.
%                       Same meaning as P in create_random_spike.
%     F                 Number of frames.
%     corr_strength     Correlation at zero distance. Scalar, or a vector of
%                       length N so that different spatial regions can have
%                       different correlation strengths. Valid range [0, 1).
%     corr_length_gSig  Correlation length expressed in units of gSig. Scalar
%                       or vector of length N. Converted internally to pixels
%                       by multiplying with gSig_fullres.
%     gSig_fullres      Neuron radius parameter in FULL-RESOLUTION pixels.
%                       Used only to convert corr_length_gSig into pixels.
%     burst_tau         Optional, default 0. Width in FRAMES of a Gaussian
%                       kernel applied along time to the latent field before
%                       thresholding. Zero reproduces the previous behaviour
%                       exactly.
%
%                       WHY IT MATTERS. Thresholding a Gaussian into isolated
%                       spikes destroys most of the latent correlation, which
%                       caps the achievable CALCIUM TRACE correlation near 0.12
%                       even at corr_strength 0.8. Smoothing the latent first
%                       makes threshold crossings CLUSTER, so the trace is
%                       dominated by a few large shared transients instead of
%                       many isolated spikes, and far more of the correlation
%                       survives. The smoothing is applied identically to every
%                       row, so corr(Z_i, Z_j) is preserved EXACTLY and the
%                       spatial structure is untouched; only the temporal
%                       character of the spiking changes.
%
%   OUTPUT
%     S                 N-by-F matrix of 0/1 doubles. Same shape and dtype as
%                       the S returned by create_random_spike.
%
%   PER-NEURON PARAMETERS
%   When corr_strength and/or corr_length_gSig are vectors, the pairwise
%   correlation is built from the two neurons' own values:
%
%       rho(i,j) = sqrt(corr_strength_i * corr_strength_j)
%                  * exp( -d_ij / (mean(corr_length_i, corr_length_j) * gSig) )
%
%   The geometric mean of the strengths keeps the resulting matrix positive
%   semi-definite in the common case (it is a Schur product of a positive
%   semi-definite exponential kernel with the rank-one matrix c*c', both of
%   which are positive semi-definite). Mixing correlation lengths across
%   neurons can in principle break positive definiteness; that case is
%   repaired below (see "POSITIVE DEFINITENESS").
%
%   METHOD (Gaussian copula)
%   1. Build the target correlation matrix Sigma, Sigma(i,j) = rho(d_ij),
%      Sigma(i,i) = 1.
%   2. Draw a latent Gaussian field Z = chol(Sigma)' * randn(N,F). Each row of
%      Z is standard normal, and corr(Z_i, Z_j) = Sigma(i,j).
%   3. Threshold every row at its OWN empirical quantile: neuron i spikes on
%      the round(P_i * F) frames with the largest Z(i,:). This is the sample
%      version of the population threshold S(i,:) = Z(i,:) > norminv(1 - P_i);
%      using the empirical quantile instead of norminv makes the marginal
%      firing rate come out exactly right in every single realisation instead
%      of only in expectation, which matters because F is short and the
%      firing rates in the simulator are low.
%
%   Because thresholding a Gaussian is a lossy operation, the achieved SPIKE
%   correlation is always BELOW the requested latent correlation rho. The
%   attenuation is monotone in corr_strength and gets stronger as the firing
%   rate gets sparser. This is expected behaviour of a copula model, not a
%   bug. The exact relationship for a pair with latent correlation r and rates
%   p_i, p_j is
%
%       corr_spike = (Phi2(z_i, z_j; r) - p_i*p_j) / sqrt(p_i(1-p_i) p_j(1-p_j))
%
%   with z = norminv(1-p) and Phi2 the bivariate normal upper-orthant
%   probability. If you need a specific achieved correlation, invert that
%   relation and pass the inflated value as corr_strength.
%
%   POSITIVE DEFINITENESS
%   Sigma is positive definite for a single global corr_strength/corr_length,
%   but per-neuron correlation lengths (or exact duplicate coordinates) can
%   make it merely positive semi-definite or slightly indefinite. If the
%   Cholesky factorisation fails, this function repairs Sigma by clipping the
%   negative eigenvalues to a small positive value, rescaling the result back
%   to a unit diagonal, and adding a small jitter to the diagonal. A warning
%   is printed when this happens, because the realised correlations will then
%   be very slightly different from the requested ones.
%
%   NOTE: N ~ 300 and F ~ 2000 makes this a few milliseconds. The cost is
%   O(N^3) for the Cholesky plus O(N*F*log F) for the row sorting.
%
%   See also CREATE_RANDOM_SPIKE.

%% ----- input handling -------------------------------------------------
X = X(:);
Y = Y(:);
P = P(:);
N = numel(X);

if numel(Y) ~= N
    error('correlated_spikes:size', 'X and Y must have the same number of elements.');
end
if numel(P) ~= N
    error('correlated_spikes:size', ...
        'P must have one entry per neuron (%d), got %d.', N, numel(P));
end
if ~isscalar(F) || F < 1 || F ~= fix(F)
    error('correlated_spikes:frames', 'F must be a positive integer.');
end
if ~isscalar(gSig_fullres) || gSig_fullres <= 0
    error('correlated_spikes:gSig', 'gSig_fullres must be a positive scalar.');
end
if nargin < 8 || isempty(burst_tau)
    burst_tau = 0;                      % default off: bit-for-bit as before
end
if ~isscalar(burst_tau) || burst_tau < 0
    error('correlated_spikes:burst', 'burst_tau must be a non-negative scalar.');
end

cs = expand_param(corr_strength, N, 'corr_strength');
cl = expand_param(corr_length_gSig, N, 'corr_length_gSig');

% A correlation of exactly 1 makes Sigma singular; clamp just below.
if any(cs < 0) || any(cs > 1)
    warning('correlated_spikes:range', ...
        'corr_strength clamped to [0, 1).');
end
cs = min(max(cs, 0), 1 - 1e-6);

if any(cl <= 0)
    error('correlated_spikes:corrLength', ...
        'corr_length_gSig must be strictly positive (it is a length scale).');
end

P = min(max(P, 0), 1);

%% ----- number of spikes per neuron ------------------------------------
% Deterministic rounding: neuron i gets exactly K(i) spikes, so
% mean(S,2) == K/F, which reproduces P up to the 1/F frame grid.
K = round(P * F);
K = min(max(K, 0), F);

%% ----- latent Gaussian field -----------------------------------------
if all(cs == 0)
    % Fast path: nothing to correlate, so skip Sigma entirely. The
    % thresholding below is still applied, so the marginal rates are exact
    % and the corr_strength = 0 case is a genuine independent-Bernoulli
    % reference for the correlated case.
    Z = randn(N, F);
else
    Sigma = build_sigma(X, Y, cs, cl, gSig_fullres);
    L = safe_chol(Sigma);          % lower-triangular factor, Sigma = L*L'
    Z = L * randn(N, F);
end

%% ----- burstiness: smooth the latent along TIME -----------------------
% Applied identically to every row, so the between-row correlation is
% unchanged; only the within-row temporal structure becomes clustered.
% Rows are re-standardised afterwards, but note the empirical-quantile
% threshold below makes the marginal rates exact regardless.
if burst_tau > 0
    h = ceil(3 * burst_tau);
    w = exp(-((-h:h).^2) / (2 * burst_tau^2));
    w = w / sqrt(sum(w.^2));            % keep unit variance
    Z = conv2(Z, w, 'same');
    Z = (Z - mean(Z, 2)) ./ (std(Z, 0, 2) + eps);
end

%% ----- threshold each row at its own empirical quantile ---------------
S = zeros(N, F);
[~, ord] = sort(Z, 2, 'descend');
for i = 1:N
    if K(i) > 0
        S(i, ord(i, 1:K(i))) = 1;
    end
end

end

% ======================================================================
function v = expand_param(v, N, name)
% Accept a scalar or a length-N vector, always return an N-by-1 vector.
v = v(:);
if isscalar(v)
    v = repmat(v, N, 1);
elseif numel(v) ~= N
    error('correlated_spikes:size', ...
        '%s must be a scalar or a vector of length N (%d), got %d.', ...
        name, N, numel(v));
end
v = double(v);
end

% ======================================================================
function Sigma = build_sigma(X, Y, cs, cl, gSig_fullres)
% Target correlation matrix of the latent Gaussian field.

% Pairwise Euclidean distance in full-resolution pixels.
D = hypot(X - X.', Y - Y.');

% Correlation length in pixels, averaged (arithmetic mean) over the pair.
Lpix = 0.5 * (cl + cl.') * gSig_fullres;

% Strength combined as the geometric mean of the two neurons' strengths.
Cpair = sqrt(cs * cs.');

Sigma = Cpair .* exp(-D ./ Lpix);

% Unit diagonal, exact symmetry, and no correlation of exactly 1 off the
% diagonal (duplicated coordinates with corr_strength -> 1 would do that).
Sigma = 0.5 * (Sigma + Sigma.');
Sigma = min(Sigma, 1 - 1e-9);
Sigma(1:size(Sigma, 1) + 1:end) = 1;
end

% ======================================================================
function L = safe_chol(Sigma)
% Lower-triangular Cholesky factor of Sigma, repairing Sigma if needed.
%
% Sigma is positive definite whenever a single global corr_strength and
% corr_length are used, because the exponential kernel is positive definite
% in any dimension. Per-neuron correlation lengths can break that, so if the
% factorisation fails we (a) clip the negative eigenvalues to a small
% positive number, (b) rescale back to a unit diagonal, and (c) add jitter
% to the diagonal, retrying with a geometrically growing jitter. The
% realised correlations after a repair are marginally lower than requested.

[R, flag] = chol(Sigma);
if flag == 0
    L = R.';
    return
end

warning('correlated_spikes:notPositiveDefinite', ...
    ['The requested correlation matrix is not positive definite ', ...
    '(this can happen with per-neuron correlation lengths). Repairing it ', ...
    'by eigenvalue clipping plus diagonal jitter; realised correlations ', ...
    'will be slightly below the requested ones.']);

% Eigenvalue clipping, then rescale rows/columns back to a unit diagonal.
Sigma = 0.5 * (Sigma + Sigma.');
[V, E] = eig(Sigma, 'vector');
E = max(E, 1e-10);
Sigma = V * diag(E) * V.';
Sigma = 0.5 * (Sigma + Sigma.');
d = sqrt(max(diag(Sigma), eps));
Sigma = Sigma ./ (d * d.');
Sigma(1:size(Sigma, 1) + 1:end) = 1;

jitter = 1e-10;
for attempt = 1:12
    [R, flag] = chol(Sigma + jitter * eye(size(Sigma)));
    if flag == 0
        L = R.';
        return
    end
    jitter = jitter * 10;
end

error('correlated_spikes:cholFailed', ...
    'Could not factorise the correlation matrix even after jittering.');
end
