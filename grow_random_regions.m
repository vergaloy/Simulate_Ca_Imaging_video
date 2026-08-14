function [masks, region_id_image, achieved_areas] = grow_random_regions(d1, d2, target_areas, seed)
% GROW_RANDOM_REGIONS  Partition a frame into random continuous regions of controlled area.
%
%   [masks, region_id_image, achieved_areas] = grow_random_regions(d1, d2, target_areas, seed)
%
% Splits a d1 x d2 field of view into R random, contiguous, irregularly shaped
% regions whose areas follow TARGET_AREAS. The regions are mutually exclusive and
% together cover every pixel of the frame. The intended use is to give each region
% its own neuron density: a sparse region needs a large area to hold a usable
% number of neurons and a dense region needs a small one, so the area of each
% region has to be controllable rather than left to chance.
%
% Method (multi source region growing)
%   1. R seed pixels are dropped at random, with a minimum separation between
%      seeds that scales with the equivalent radius of the two regions involved,
%      so that no region is strangled at birth by a neighbour sitting on top of it.
%   2. All regions grow simultaneously. In every round each region that has not
%      yet reached its target area claims one more pixel, drawn AT RANDOM from its
%      current frontier of unclaimed 4 connected neighbours. Drawing at random
%      rather than in first in first out order is what makes the boundaries
%      irregular and free of straight axis aligned edges.
%   3. A region stops expanding as soon as it reaches its target area.
%   4. Pixels still unclaimed at the end (pockets sealed off when the regions
%      around them stopped growing) are handed over whole, one connected pocket at
%      a time, to the adjacent region with the largest remaining area deficit.
%      Assigning a whole connected pocket to a region that touches it is what keeps
%      every region a single connected component.
%
% Inputs
%   d1, d2        frame height (rows) and width (columns), in pixels. The masks
%                 are returned in the same [row, column] convention used by
%                 GET_RANDOM_POINTS_REGIONS.
%   target_areas  1xR vector of desired areas, in pixels. Because the regions must
%                 jointly cover the frame, only the RATIOS between the entries can
%                 be honoured: the values are rescaled proportionally so that they
%                 sum to exactly d1*d2. A warning is issued when the rescaling
%                 factor differs from 1 by more than 1 per cent, so that a caller
%                 who meant the numbers literally is told about it.
%   seed          optional, scalar seed for the random number generator. The
%                 previous global random state is restored on exit, so calling this
%                 function does not disturb the randomness of the rest of the
%                 simulation.
%
% Outputs
%   masks            1xR cell of logical d1 x d2 masks, mutually exclusive and
%                    jointly covering the frame. Feeds straight into
%                    GET_RANDOM_POINTS_REGIONS.
%   region_id_image  d1 x d2 double image, entry r where region r owns the pixel.
%   achieved_areas   1xR vector with the number of pixels actually assigned to
%                    each region.
%
% Example
%   [masks, ids, areas] = grow_random_regions(320, 320, [47000 19000 9400 6700], 7);
%   imagesc(ids); axis image
%   [X, Y, rid] = get_random_points_regions(masks, [60 60 60 60], [12 8 6 5]);
%
% See also GET_RANDOM_POINTS_REGIONS, GET_RANDOM_POINTS.

if nargin < 4
    seed = [];
end

% ---------------------------------------------------------------- validation
validateattributes(d1, {'numeric'}, {'scalar', 'integer', 'positive'}, mfilename, 'd1');
validateattributes(d2, {'numeric'}, {'scalar', 'integer', 'positive'}, mfilename, 'd2');
validateattributes(target_areas, {'numeric'}, {'vector', 'positive', 'finite'}, mfilename, 'target_areas');

d1   = double(d1);
d2   = double(d2);
npix = d1 * d2;
nreg = numel(target_areas);

if nreg > npix
    error('grow_random_regions:tooManyRegions', ...
        'Asked for %d regions in a frame of only %d pixels.', nreg, npix);
end

% ------------------------------------------------- random state, seeded here
if ~isempty(seed)
    previous_state = rng;
    restore_state  = onCleanup(@() rng(previous_state));
    rng(seed, 'twister');
end

% ------------------------------------------- rescale the targets to the frame
requested = double(target_areas(:)');
scale     = npix / sum(requested);
if abs(scale - 1) > 0.01
    warning('grow_random_regions:targetsRescaled', ...
        ['The requested areas sum to %d but the frame holds %d pixels. The areas ' ...
         'were rescaled by %.4f so that the regions cover the frame; only their ' ...
         'ratios are preserved.'], round(sum(requested)), npix, scale);
end
target = local_round_to_total(requested * scale, npix);

% ------------------------------------------------------------ seed placement
seed_px = local_place_seeds(d1, d2, target);

% ------------------------------------------------------------------- growing
labels = zeros(npix, 1);          % 0 = unclaimed, r = owned by region r
queued = false(npix, nreg);       % pixel already sitting in the frontier of r
counts = zeros(1, nreg);

frontier = cell(1, nreg);
flen     = zeros(1, nreg);
for r = 1:nreg
    frontier{r} = zeros(1024, 1);
end

% The frontier bookkeeping is written out inline rather than in a helper: QUEUED
% is a large array and handing it to a function that modifies it would force a
% full copy on every single pixel claimed.

% Plant the seeds. A collision can only happen if two seeds landed on the same
% pixel, which the separation rule makes impossible unless it had to be relaxed
% all the way down; the check keeps the bookkeeping honest in that corner case.
for r = 1:nreg
    p = seed_px(r);
    if labels(p) == 0
        labels(p) = r;
        counts(r) = 1;
        nb = local_new_neighbours(p, r, d1, d2, npix, labels, queued);
        queued(nb + (r - 1) * npix) = true;
        [frontier{r}, flen(r)] = local_append(frontier{r}, flen(r), nb);
    end
end

% Each region claims pixels at a rate proportional to its target area, so that
% every region reaches its target at the same moment. Growing all regions at the
% SAME absolute rate instead would be a trap: the small regions would reach their
% target early, freeze, and wall the large region in long before it had collected
% its area, which shows up as a large region 20 per cent short and its neighbour
% swollen by the leftovers. With proportional rates the region radii stay in the
% ratio sqrt(target) throughout, which is the ratio the seed spacing assumes.
rate   = target / max(target);
credit = zeros(1, nreg);

while true
    active = find(counts < target & flen > 0);
    if isempty(active)
        break
    end
    credit(active) = credit(active) + rate(active);
    active = active(randperm(numel(active)));   % no region gets a systematic edge
    progress  = false;
    attempted = false;

    for k = 1:numel(active)
        r = active(k);
        quota     = floor(credit(r));
        credit(r) = credit(r) - quota;
        quota     = min(quota, target(r) - counts(r));
        attempted = attempted || quota > 0;

        claimed = 0;
        % Pop random frontier entries until one of them is still unclaimed.
        % Popping at random, not in queue order, is what roughens the borders.
        while claimed < quota && flen(r) > 0
            j = randi(flen(r));
            p = frontier{r}(j);
            frontier{r}(j)   = frontier{r}(flen(r));
            flen(r)          = flen(r) - 1;
            if labels(p) ~= 0
                continue
            end
            labels(p) = r;
            counts(r) = counts(r) + 1;
            claimed   = claimed + 1;
            nb = local_new_neighbours(p, r, d1, d2, npix, labels, queued);
            queued(nb + (r - 1) * npix) = true;
            [frontier{r}, flen(r)] = local_append(frontier{r}, flen(r), nb);
            progress = true;
        end
    end

    % A round in which nobody even had a whole pixel of credit is not a stall,
    % the small regions simply have to wait for their credit to build up.
    if attempted && ~progress
        break
    end
end

% ------------------------------------------------- hand out the leftover pockets
leftover = (labels == 0);
if any(leftover)
    cc    = bwconncomp(reshape(leftover, d1, d2), 4);
    sizes = cellfun(@numel, cc.PixelIdxList);
    [~, order] = sort(sizes, 'descend');
    for k = order(:)'
        px  = cc.PixelIdxList{k};
        nb  = local_neighbour_indices(px, d1, d2);
        adj = unique(labels(nb));
        adj = adj(adj > 0);
        deficit = target - counts;
        if isempty(adj)
            [~, r] = max(deficit);          % unreachable in practice
        else
            [~, i] = max(deficit(adj));
            r = adj(i);
        end
        labels(px) = r;
        counts(r)  = counts(r) + numel(px);
    end
end

% --------------------------------------------------------------- area balancing
% A region can be walled in by its neighbours before it has collected its target
% area, which leaves it short and leaves somebody else swollen. Repair that by
% pushing the border between them: a region that is short takes pixels along its
% border from a neighbour that is better off than it is. A pixel is only taken
% when its removal cannot break the donor apart, which is decided by a local test
% on the 3x3 neighbourhood (the S pixels around it must form a single contiguous
% arc), so both regions stay a single connected component. The receiving region
% only ever gains a pixel that already touches it, so it stays connected too.
%
% The donor test compares RELATIVE surplus (counts - target)/target, and a region
% sitting exactly on target is therefore a legitimate donor to a region that is
% short. That is what makes the repair work when the swollen region does not
% touch the starved one: the pixels travel in a chain, each region in turn giving
% up a pixel to its poorer neighbour and taking one back from its richer one.
% Restricting donors to regions already over target instead leaves a starved
% region unrepaired whenever the surplus sits somewhere else in the frame, which
% is the common case with four regions of very different size.
%
% Termination: a move is accepted only when it strictly decreases the imbalance
% f = sum((counts - target).^2 ./ target). Moving one pixel from S to D changes f
% by 2*(score(D) - score(S)) + 1/target(D) + 1/target(S), so requiring the score
% gap to exceed half the sum of the reciprocals makes every accepted move a
% strict decrease of a quantity bounded below.
tol_rel = 0.002;                       % aim an order of magnitude inside the 5 %
plus_se = [0 1 0; 1 1 1; 0 1 0];

for pass = 1:2000
    score = (counts - target) ./ target;
    if -min(score) <= tol_rel
        break                                    % nobody is meaningfully short
    end
    [~, poorest_first] = sort(score, 'ascend');
    progressed = false;

    for oi = 1:nreg
        D = poorest_first(oi);
        if (target(D) - counts(D)) / target(D) <= tol_rel
            break                                % the rest are not short either
        end

        maskD  = reshape(labels == D, d1, d2);
        border = imdilate(maskD, plus_se) & ~maskD;
        cand   = find(border);
        if isempty(cand)
            continue
        end
        cand = cand(randperm(numel(cand)));

        for k = 1:numel(cand)
            p = cand(k);
            S = labels(p);
            gap = (counts(S) - target(S)) / target(S) - (counts(D) - target(D)) / target(D);
            if gap <= 0.5 * (1 / target(D) + 1 / target(S))
                continue                         % move would not reduce the imbalance
            end
            if ~local_is_removable(p, S, labels, d1, d2)
                continue                         % would split the donor in two
            end
            labels(p)  = D;
            counts(D)  = counts(D) + 1;
            counts(S)  = counts(S) - 1;
            progressed = true;
            if (target(D) - counts(D)) / target(D) <= tol_rel
                break
            end
        end
    end

    if ~progressed
        break                                    % no legal move left anywhere
    end
end

% -------------------------------------------------------------------- outputs
region_id_image = reshape(labels, d1, d2);
masks           = cell(1, nreg);
achieved_areas  = zeros(1, nreg);
for r = 1:nreg
    masks{r}          = (region_id_image == r);
    achieved_areas(r) = nnz(masks{r});
end

% Cheap guarantee check: growth from a single seed into unclaimed pixels and
% pocket assignment by whole connected component both preserve connectivity, so
% a failure here means the algorithm was changed and broke that property.
for r = 1:nreg
    if achieved_areas(r) == 0
        warning('grow_random_regions:emptyRegion', ...
            'Region %d ended up empty; its target area was probably too small.', r);
    elseif bwconncomp(masks{r}, 4).NumObjects > 1
        warning('grow_random_regions:disconnectedRegion', ...
            'Region %d is not a single connected component.', r);
    end
end
end

% ============================================================== local functions

function out = local_round_to_total(x, total)
% Round a vector of real targets to integers that sum exactly to TOTAL
% (largest remainder method), keeping every entry at least 1.
out       = max(1, floor(x));
remainder = total - sum(out);
if remainder > 0
    [~, order] = sort(x - floor(x), 'descend');
    for k = 1:remainder
        i = order(mod(k - 1, numel(x)) + 1);
        out(i) = out(i) + 1;
    end
elseif remainder < 0
    % Only possible when the floor of tiny targets was lifted to 1; take the
    % excess back from the largest regions, where it is negligible.
    for k = 1:(-remainder)
        [~, i] = max(out);
        out(i) = out(i) - 1;
    end
end
end

function seed_px = local_place_seeds(d1, d2, target)
% Drop one seed per region, big regions first, keeping seeds apart by a distance
% that scales with the equivalent radii of the two regions involved, and keeping
% each seed a little away from the frame edge so that a region does not waste
% half of its growth pushing against the border.
nreg   = numel(target);
radius = sqrt(target / pi);
[~, order] = sort(target, 'descend');

% Margin, per region, capped so it can never swallow the frame.
margin = min(round(0.35 * radius), floor(0.2 * min(d1, d2)));

alpha   = 0.75;
seed_px = zeros(1, nreg);
for restart = 1:40
    rows = zeros(1, nreg);
    cols = zeros(1, nreg);
    ok_all = true;
    for k = 1:nreg
        r = order(k);
        placed = false;
        lo1 = 1 + margin(r);  hi1 = d1 - margin(r);
        lo2 = 1 + margin(r);  hi2 = d2 - margin(r);
        for attempt = 1:2000
            rr = randi([lo1 hi1]);
            cc = randi([lo2 hi2]);
            good = true;
            for m = 1:(k - 1)
                q = order(m);
                need = alpha * (radius(r) + radius(q));
                if hypot(rr - rows(q), cc - cols(q)) < need
                    good = false;
                    break
                end
            end
            if good
                rows(r) = rr;
                cols(r) = cc;
                placed  = true;
                break
            end
        end
        if ~placed
            ok_all = false;
            break
        end
    end
    if ok_all
        seed_px = rows + (cols - 1) * d1;
        return
    end
    % Frame too tight for this spacing: relax both constraints and retry.
    alpha  = alpha * 0.85;
    margin = floor(margin * 0.85);
end

% Last resort: unconstrained distinct pixels.
warning('grow_random_regions:seedSpacing', ...
    'Could not honour a minimum seed separation; seeds were placed without one.');
p = randperm(d1 * d2, nreg);
seed_px = p;
end

function nb = local_new_neighbours(p, r, d1, d2, npix, labels, queued)
% The 4 connected neighbours of pixel P that are still unclaimed and not already
% waiting in the frontier of region R. Reads its inputs only, never writes them,
% so no large array is copied.
row = mod(p - 1, d1) + 1;
col = (p - row) / d1 + 1;

nb = zeros(4, 1);
n  = 0;
if row > 1,  n = n + 1; nb(n) = p - 1;  end
if row < d1, n = n + 1; nb(n) = p + 1;  end
if col > 1,  n = n + 1; nb(n) = p - d1; end
if col < d2, n = n + 1; nb(n) = p + d1; end
nb = nb(1:n);
nb = nb(labels(nb) == 0 & ~queued(nb + (r - 1) * npix));
end

function [f, n] = local_append(f, n, v)
m = numel(v);
if m == 0
    return
end
while n + m > numel(f)
    f = [f; zeros(max(numel(f), m), 1)];   % amortised doubling
end
f(n + 1 : n + m) = v;
n = n + m;
end

function tf = local_is_removable(p, S, labels, d1, d2)
% True when pixel P can be taken away from region S without splitting S in two.
% Local test: walk the 8 pixels around P as a cycle (each one is 4 adjacent to
% the next), and require the ones belonging to S to form a single contiguous arc.
% Anything outside the frame counts as not belonging to S. The test is
% conservative, it can refuse a pixel that would in fact have been safe, which
% only costs a little area accuracy and never costs connectivity.
row = mod(p - 1, d1) + 1;
col = (p - row) / d1 + 1;

up = row > 1;  dn = row < d1;  lf = col > 1;  rt = col < d2;

v = false(1, 8);                        % N NE E SE S SW W NW
if up,       v(1) = labels(p - 1)      == S; end
if up && rt, v(2) = labels(p - 1 + d1) == S; end
if rt,       v(3) = labels(p + d1)     == S; end
if dn && rt, v(4) = labels(p + 1 + d1) == S; end
if dn,       v(5) = labels(p + 1)      == S; end
if dn && lf, v(6) = labels(p + 1 - d1) == S; end
if lf,       v(7) = labels(p - d1)     == S; end
if up && lf, v(8) = labels(p - 1 - d1) == S; end

if ~any(v)
    tf = false;                          % P is an isolated pixel of S
    return
end
narcs = sum(v & ~v([8 1:7]));            % 0 when the whole ring belongs to S
tf = narcs <= 1;
end

function nb = local_neighbour_indices(px, d1, d2)
% All 4 connected neighbours of a list of linear indices, as a column vector
% (duplicates and the pixels themselves may appear, the caller only takes
% unique labels from it).
px  = px(:);
row = mod(px - 1, d1) + 1;
col = (px - row) / d1 + 1;
nb  = [px(row > 1)  - 1; ...
       px(row < d1) + 1; ...
       px(col > 1)  - d1; ...
       px(col < d2) + d1];
end

