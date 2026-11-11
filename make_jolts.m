function J = make_jolts(T, p, seed, avoid)
%% Jolts: a fast displacement of the SAME scene, as when the animal knocks the scope.
%
% Unlike an impact (make_impacts), where the view is lost to a foreign field
% of view, a jolt keeps the scene: the tissue moves a few pixels fast, with
% motion blur, and comes back more slowly. On the way it goes a little out of
% focus and its brightness changes a little. The baseline's low-pass part
% (the vignetting) stays fixed to the optics. apply_camera_motion builds the
% frames (jolt_frame); this draws the jolts and their time course, as plain
% numbers.
%
% Time course of one jolt starting at frame t0. Frame k is exposed over
% [k - 1/2, k + 1/2]; u = tau - (t0 - 1/2) is the time since the jolt began,
% in frames. With rise n_r, return n_b (both in frames, not whole):
%
%   q(u) = ss(u / n_r)                    0 <= u < n_r      fast rise
%   q(u) = 1 - ss((u - n_r) / n_b)        n_r <= u <= n_r + n_b   slower return
%   ss(x) = (1 - cos(pi * clip(x, 0, 1))) / 2
%
%   offset(u)  = size * q(u) * [cos angle, sin angle]
%   defocus(t) = defocus_peak * mean q        Gaussian sigma, px
%   gain(t)    = 1 + brightness * mean q      brightness signed
%
% Motion blur: each frame averages n_sub sub-frame instants of its exposure,
% as many as the offset's path within the exposure needs for one instant per
% blur_step px (at most blur_max). The means are over those instants, so the
% ground truth is what the blurred frame shows.
%
% Everything is drawn from a stream of its own, so the rest of the simulation
% is the same with or without jolts.
%
% Inputs:
%   T     - number of frames
%   p     - struct, the jolt_* options without the prefix (see
%           Simulate_Ca_video): n, onset, size, rise, return_frames, defocus,
%           brightness, blur_step, blur_max, gap
%   seed  - seed of the stream
%   avoid - optional [1 T] logical: frames a jolt must keep gap away from
%           (the impacts)
%
% Output, struct J:
%   table [n 7]   one row per jolt: [t0 rise return size angle defocus
%                 brightness]; angle in radians (x = column, y = row)
%   cols          names of the table columns
%   index [1 T]   which jolt each frame belongs to (0 = none)
%   mask  [1 T]   index > 0
%   offset_xy [2 T]  the displacement, averaged over the exposure, px
%   profile [1 T]    mean q over the exposure
%   defocus [1 T], gain [1 T]
%   n_sub [1 T]      sub-frame instants of each frame (0 outside jolts)
%   sub_x, sub_y [blur_max T]  their offsets (NaN past n_sub)
%
% Author: Pablo Vergara

if nargin < 4 || isempty(avoid)
    avoid = false(1, T);
end
nmax = max(1, round(p.blur_max));
J = struct();
J.cols = {'t0', 'rise', 'return', 'size', 'angle', 'defocus', 'brightness'};
J.table = zeros(0, 7);
J.index = zeros(1, T);
J.mask = false(1, T);
J.offset_xy = zeros(2, T);
J.profile = zeros(1, T);
J.defocus = zeros(1, T);
J.gain = ones(1, T);
J.n_sub = zeros(1, T);
J.sub_x = nan(nmax, T); J.sub_y = nan(nmax, T);

rs = RandStream('mt19937ar', 'Seed', seed);
if isempty(p.onset)
    n = round(p.n);
    if n == 0
        return
    end
else
    n = numel(p.onset);
end
% Draw every jolt's shape first, then place them.
rise = p.rise(1) + rand(rs, 1, n) * diff(p.rise([1 end]));
ret = p.return_frames(1) + rand(rs, 1, n) * diff(p.return_frames([1 end]));
sz = p.size(1) + rand(rs, 1, n) * diff(p.size([1 end]));
ang = 2 * pi * rand(rs, 1, n);
dfc = p.defocus(1) + rand(rs, 1, n) * diff(p.defocus([1 end]));
brt = (p.brightness(1) + rand(rs, 1, n) * diff(p.brightness([1 end]))) .* ...
    (1 - 2 * (rand(rs, 1, n) < 0.5));
len = ceil(rise + ret);                            % frames each jolt touches

if isempty(p.onset)
    gap = p.gap;
    % Frames a jolt may not touch: the ends and the avoided frames, padded by gap.
    bad = conv(double(avoid), ones(1, 2 * gap + 1), 'same') > 0;
    bad(1:gap) = true; bad(end - gap + 1:end) = true;
    t0 = zeros(1, n);
    for tries = 1:10000
        ok = true;
        used = bad;
        for j = 1:n
            cand = find(~used(1:T - len(j) + 1));
            cand = cand(arrayfun(@(a) ~any(used(a:a + len(j) - 1)), cand));
            if isempty(cand), ok = false; break; end
            t0(j) = cand(1 + floor(rand(rs) * numel(cand)));
            used(max(1, t0(j) - gap):min(T, t0(j) + len(j) - 1 + gap)) = true;
        end
        if ok, break; end
    end
    assert(ok, 'Simulate_Ca_video:jolts', 'could not place %d jolts in %d frames', n, T);
else
    t0 = round(p.onset(:)');
    for j = 1:n
        assert(t0(j) >= 1 && t0(j) + len(j) - 1 <= T, 'Simulate_Ca_video:jolts', 'jolt outside the movie');
    end
end
[t0, order] = sort(t0);
rise = rise(order); ret = ret(order); sz = sz(order); ang = ang(order);
dfc = dfc(order); brt = brt(order); len = len(order);
assert(n == 1 || all(t0(2:end) >= t0(1:end-1) + len(1:end-1)), 'Simulate_Ca_video:jolts', 'jolts overlap');

ss = @(x) (1 - cos(pi * min(max(x, 0), 1))) / 2;
q = @(u, a, b) (u < a) .* ss(u / a) + (u >= a) .* (1 - ss((u - a) / b));
fine = ((1:64)' - 0.5) / 64 - 0.5;                % to measure the path
for j = 1:n
    J.table(end+1, :) = [t0(j) rise(j) ret(j) sz(j) ang(j) dfc(j) brt(j)];
    for k = t0(j):t0(j) + len(j) - 1
        uf = k + fine - (t0(j) - 0.5);
        qf = q(uf, rise(j), ret(j));
        path = sz(j) * sum(abs(diff(qf)));
        m = min(nmax, max(1, ceil(path / p.blur_step)));
        tau = ((1:m)' - 0.5) / m - 0.5;
        qk = q(k + tau - (t0(j) - 0.5), rise(j), ret(j));
        J.n_sub(k) = m;
        J.sub_x(1:m, k) = sz(j) * qk * cos(ang(j));
        J.sub_y(1:m, k) = sz(j) * qk * sin(ang(j));
        J.offset_xy(:, k) = [mean(J.sub_x(1:m, k)); mean(J.sub_y(1:m, k))];
        J.profile(k) = mean(qk);
        J.defocus(k) = dfc(j) * J.profile(k);
        J.gain(k) = 1 + brt(j) * J.profile(k);
        J.index(k) = j;
    end
end
J.mask = J.index > 0;
end
