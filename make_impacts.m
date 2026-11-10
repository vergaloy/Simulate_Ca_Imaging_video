function I = make_impacts(T, p, seed)
%% Impacts: the animal knocks the microscope and the view is lost for about 1 s.
%
% During an impact the tissue's structure (the baseline's high-pass part, the
% neurons and the background sources) slides away fast, with motion blur, and
% blends into the high-pass structure of a different field of view (another
% baseline image); at the peak only that foreign structure is visible. Then it
% reverses: the foreign structure blends back out and the real scene slides
% back. The baseline's low-pass part (vignetting, broad brightness) is fixed to
% the optics and never moves. apply_camera_motion builds the frames; this draws
% the impacts and their time course, as plain numbers.
%
% Time course of one impact, starting at frame t0 and lasting D frames. Frame k
% is exposed over [k - 1/2, k + 1/2]; u = tau - (t0 - 1/2) is the time since
% the impact began, in frames, so the impact spans u in [0, D]:
%
%   slide(u) = dist * min(ss(u / n_slide), ss((D - u) / n_slide))
%   w(u)     = 1 - min(ss(u / n_blend), ss((D - u) / n_blend))
%   ss(x)    = (1 - cos(pi * clip(x, 0, 1))) / 2
%
% slide is along the impact's direction; w is the fraction of the real scene
% still visible (1 - w is the foreign structure's). Each frame averages
% blur_samples sub-frame instants of its exposure (motion blur). The foreign
% structure moves with the slide, offset so that it is at rest, unshifted, at
% the peak: slide(u) - dist.
%
% Everything is drawn from a stream of its own, so the rest of the simulation
% is the same with or without impacts.
%
% Inputs:
%   T    - number of frames
%   p    - struct: n (impacts; used when onset is empty), onset (frames, [] =
%          random), duration (frames), distance ([min max] px), slide_frames,
%          blend_frames, blur_samples, gap (frames kept clear between impacts
%          and from the movie's ends when the onsets are random)
%   seed - seed of the stream
%
% Output, struct I:
%   table   [n 4]   one row per impact: [t0 duration distance angle]; angle in
%                   radians (x = column, y = row)
%   cols            names of the table columns
%   index   [1 T]   which impact each frame belongs to (0 = none)
%   mask    [1 T]   index > 0
%   weight  [1 T]   fraction of the real scene visible, averaged over the
%                   exposure (1 outside impacts)
%   slide_xy [2 T]  the real scene's slide, averaged over the exposure, px
%                   (what the blurred frame shows; 0 outside impacts)
%   sub_x, sub_y, sub_w [nb T]  the slide and weight at each sub-frame instant
%   sub_fx, sub_fy      [nb T]  the foreign structure's offset at each instant
%
% Author: Pablo Vergara

nb = max(1, round(p.blur_samples));
D = round(p.duration);
I = struct();
I.cols = {'t0', 'duration', 'distance', 'angle'};
I.table = zeros(0, 4);
I.index = zeros(1, T);
I.mask = false(1, T);
I.weight = ones(1, T);
I.slide_xy = zeros(2, T);
I.sub_x = zeros(nb, T); I.sub_y = zeros(nb, T); I.sub_w = ones(nb, T);
I.sub_fx = zeros(nb, T); I.sub_fy = zeros(nb, T);

rs = RandStream('mt19937ar', 'Seed', seed);
if isempty(p.onset)
    n = round(p.n);
    if n == 0
        return
    end
    gap = p.gap;
    lo = 1 + gap; hi = T - D + 1 - gap;           % onsets keep gap from both ends
    assert(hi - lo >= (n - 1) * (D + gap), 'Simulate_Ca_video:impacts', ...
        '%d impacts of %d frames, %d apart, do not fit in %d frames', n, D, gap, T);
    for tries = 1:10000
        t0 = sort(lo + floor(rand(rs, 1, n) * (hi - lo + 1)));
        if n == 1 || all(diff(t0) >= D + gap), break; end
    end
    assert(n == 1 || all(diff(t0) >= D + gap), 'Simulate_Ca_video:impacts', 'could not space the impacts');
else
    t0 = sort(round(p.onset(:)'));
    n = numel(t0);
    assert(all(t0 >= 1) && all(t0 + D - 1 <= T), 'Simulate_Ca_video:impacts', 'impact outside the movie');
    assert(n == 1 || all(diff(t0) >= D), 'Simulate_Ca_video:impacts', 'impacts overlap');
end

ss = @(x) (1 - cos(pi * min(max(x, 0), 1))) / 2;
tau = ((1:nb)' - 0.5) / nb - 0.5;                 % sub-frame instants, [nb 1]
for j = 1:n
    dist = p.distance(1) + rand(rs) * diff(p.distance([1 end]));
    ang = 2*pi*rand(rs);
    I.table(end+1, :) = [t0(j) D dist ang];
    for k = t0(j):t0(j) + D - 1
        u = k + tau - (t0(j) - 0.5);
        a = dist * min(ss(u / p.slide_frames), ss((D - u) / p.slide_frames));
        w = 1 - min(ss(u / p.blend_frames), ss((D - u) / p.blend_frames));
        I.sub_x(:, k) = a * cos(ang); I.sub_y(:, k) = a * sin(ang); I.sub_w(:, k) = w;
        I.sub_fx(:, k) = (a - dist) * cos(ang); I.sub_fy(:, k) = (a - dist) * sin(ang);
        I.index(k) = j;
    end
end
I.mask = I.index > 0;
I.weight = mean(I.sub_w, 1);
I.slide_xy = [mean(I.sub_x, 1); mean(I.sub_y, 1)];
end
