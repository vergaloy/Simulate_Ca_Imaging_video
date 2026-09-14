function [n, info] = density_to_count(k_neighbours, area_px, gSig_fullres, min_dist)
%DENSITY_TO_COUNT  Neurons needed in an area to hit a target local density.
%
%   [n, info] = density_to_count(k_neighbours, area_px, gSig_fullres, min_dist)
%
%   Density is specified the way it is interpretable: k = the expected number
%   of OTHER neurons within 4*gSig of a given neuron. For a homogeneous
%   Poisson field of surface density lambda,
%
%       k = lambda * pi * (4*gSig)^2      =>   lambda = k / (pi*(4*gSig)^2)
%       n = lambda * area
%
%   PACKING CHECK. A hard-core minimum separation caps how many points fit:
%   the densest possible arrangement is hexagonal, giving at most
%   area / (min_dist^2 * sqrt(3)/2) points, and random sequential placement
%   saturates well below that. `info.feasible` reports n against a
%   conservative 60% of the hexagonal limit.
%
%   This check exists because the previous simulation silently under-filled:
%   nominal k of 5 came out as a realised 3.31 because the placement routine
%   ran out of admissible positions and returned fewer points than asked.

lambda = k_neighbours / (pi * (4 * gSig_fullres)^2);
n = round(lambda * area_px);

hex_limit = area_px / (min_dist^2 * sqrt(3)/2);
safe_limit = floor(0.60 * hex_limit);

info = struct('lambda', lambda, 'n_requested', n, ...
              'hex_limit', hex_limit, 'safe_limit', safe_limit, ...
              'feasible', n <= safe_limit, ...
              'realised_k_if_capped', ...
                  min(n, safe_limit) / area_px * pi * (4*gSig_fullres)^2);

if ~info.feasible
    warning('density_to_count:packing', ...
        ['k=%.1f needs %d neurons in %d px but min_dist %.2f px allows only ~%d. ' ...
         'Realised density would be k=%.2f, not %.2f. Enlarge the area or ' ...
         'reduce min_dist.'], k_neighbours, n, round(area_px), min_dist, ...
        safe_limit, info.realised_k_if_capped, k_neighbours);
end
end
