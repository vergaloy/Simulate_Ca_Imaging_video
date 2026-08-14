function [X, Y, region_id] = get_random_points_regions(region_masks, n_per_region, min_dist_per_region, plotme)
% GET_RANDOM_POINTS_REGIONS  Place points with a different density per region.
%
%   [X, Y, region_id] = get_random_points_regions(region_masks, n_per_region, ...
%                                                 min_dist_per_region, plotme)
%
% Generalisation of GET_RANDOM_POINTS to several spatial regions, each with its
% own target number of points and its own minimum separation. Regions are filled
% in the order they are given, and a candidate point is accepted only when it is
% at least the minimum separation OF ITS OWN REGION away from EVERY point that
% has already been placed (including points of earlier regions). Enforcing the
% constraint across regions is what stops a dense clump and the sparse
% background from piling on top of each other along the region border.
%
% Inputs
%   region_masks        1xR cell of logical d1 x d2 masks. They should be
%                       mutually exclusive; overlap is not checked here, the
%                       caller is expected to have resolved it.
%   n_per_region        1xR number of points requested in each region.
%   min_dist_per_region 1xR minimum separation (pixels) enforced for each region.
%   plotme              optional, scalar flag, plots the resulting points.
%
% Outputs
%   X, Y        1xN row vectors of column (X) and row (Y) pixel coordinates, in
%               the coordinate frame of the masks. Same orientation and class as
%               the output of GET_RANDOM_POINTS.
%   region_id   1xN index of the region each point was placed in.
%
% A region that cannot be filled (minimum separation too large for the area
% available) issues a warning and returns fewer points than requested rather
% than looping forever.

if nargin < 4 || isempty(plotme)
    plotme = 0;
end

R = numel(region_masks);
if numel(n_per_region) ~= R || numel(min_dist_per_region) ~= R
    error('get_random_points_regions:badSize', ...
        'region_masks, n_per_region and min_dist_per_region must have the same length.');
end

X = [];
Y = [];
region_id = [];

for r = 1:R
    n  = n_per_region(r);
    md = min_dist_per_region(r);
    if n <= 0
        continue
    end

    [yy, xx] = find(region_masks{r});
    if isempty(yy)
        warning('get_random_points_regions:emptyRegion', ...
            'Region %d contains no allowed pixel, no point placed there.', r);
        continue
    end

    % Same shuffling strategy as get_random_points: walk a random permutation
    % of the candidate pixels and keep the ones that clear the distance test.
    p  = randperm(numel(yy), numel(yy));
    yy = yy(p);
    xx = xx(p);

    added = 0;
    for k = 1:numel(xx)
        thisX = xx(k);
        thisY = yy(k);
        if isempty(X)
            ok = true;
        else
            ok = min(sqrt((thisX - X).^2 + (thisY - Y).^2)) >= md;
        end
        if ok
            X(end+1) = thisX; %#ok<AGROW>
            Y(end+1) = thisY; %#ok<AGROW>
            region_id(end+1) = r; %#ok<AGROW>
            added = added + 1;
            if added >= n
                break
            end
        end
    end

    if added < n
        warning('get_random_points_regions:underfilled', ...
            ['Region %d: %d points requested, %d placed. The minimum separation ' ...
             '(%.4g px) is too large for the area available.'], r, n, added, md);
    end
end

if plotme
    plot(X, Y, 'b*');
    set(gca, 'YDir', 'reverse');
end
end
