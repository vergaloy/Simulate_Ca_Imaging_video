function Bu = make_bursts(C, xy, frame_size, p, seed)
%% Activity bursts: periods of strong, synchronised, epileptic-like activity.
%
% During a burst many neurons fire together for seconds and the diffuse
% background (out-of-focus light, the neuropil) rises with them. The image
% changes a lot, with no motion at all. This draws the bursts and returns the
% activity they add to the neurons, as traces on top of the neurons' own (C);
% Simulate_Ca_video turns them into the image on the movie's own grey scale,
% adds the glow (the in-focus burst image blurred, see glow_ratio), and so
% leaves the frames outside bursts as they would be without them.
%
% One burst, starting at frame t0 and driving for D frames:
%
%   drive(u) = min(ss(u / ramp), ss((D - u) / ramp)),  u = t - t0 + 1/2
%   ss(x)    = (1 - cos(pi * clip(x, 0, 1))) / 2
%
% Where: a centre drawn uniformly in the frame and a radius R; a neuron at
% distance d from the centre is recruited with probability
% fraction * exp(-d^2 / (2 R^2)) (R = Inf: fraction everywhere).
% When: the population discharges rhythmically while the drive is on, at t0
% plus a running sum of intervals drawn from p.interval (frames). Each
% recruited neuron joins each discharge with its own probability (drawn from
% p.join) times the drive, so the burst builds up and dies down, and fires
% d / speed frames after the discharge (speed = Inf: all at once; finite: a
% wave spreading out from the centre). Its spikes become calcium with a kernel
% drawn from the simulator's own rise and decay distributions, scaled so that
% its peak is p.amplitude (drawn per neuron) times the neuron's own largest
% transient in the session (max of C): 1 is as bright as its brightest normal
% transient.
%
% Every [min max] option may instead have one row per burst (in time order),
% to give each burst its own range.
%
% Everything is drawn from a stream of its own, so the rest of the simulation
% is the same with or without bursts.
%
% Inputs:
%   C          - [N T] the neurons' traces in the session
%   xy         - [N 2] their centres (x, y) in the saved frame, px
%   frame_size - [d1 d2] of the saved frame
%   p          - struct: n (bursts; used when onset is empty), onset (first
%                frames, [] = random), duration ([min max] drive frames), gap
%                (frames kept clear between random bursts and from the ends),
%                ramp (frames the drive takes to rise and fall), and drawn per
%                burst: fraction, radius (px), speed (px per frame), neuropil
%                (glow ratio); per neuron: amplitude, join; interval (frames
%                between discharges); tail (the burst ends where its
%                population trace falls below this fraction of its peak, and
%                adds nothing after); rise, decay (the simulator's lognormal
%                parameters of 1/rise and 1/decay, as create_neuron_data uses)
%   seed       - seed of the stream
%
% Output, struct Bu:
%   table      [n 11] one row per burst, columns in cols: t0 (first driven
%              frame), duration (drive frames), n_recruited, fraction (of all
%              neurons), last (last frame: the population trace has decayed
%              below p.tail of its peak), peak (frame where it peaks), cx, cy
%              (centre, saved-frame px, 1-based), radius, speed, neuropil
%              (glow ratio); frames 1-based
%   recruited  [N n] logical: neuron took part in burst j
%   amp        [N n] its amplitude in burst j (0 if not recruited)
%   delay      [N n] its delay after each discharge, frames (wave)
%   index      [1 T] which burst each frame belongs to (0 = none)
%   mask       [1 T] index > 0
%   level      [1 T] the recruited neurons' mean burst trace, peak 1 per
%              burst, 0 outside bursts
%   glow_ratio [1 T] the burst's glow ratio on its frames, 0 elsewhere: the
%              glow's total light over the in-focus burst image's
%   C_burst    [N T] activity added to the neurons
%
% Author: Pablo Vergara

[N, T] = size(C);
Bu = struct();
Bu.cols = {'t0', 'duration', 'n_recruited', 'fraction', 'last', 'peak', ...
    'cx', 'cy', 'radius', 'speed', 'neuropil'};
Bu.table = zeros(0, numel(Bu.cols));
Bu.recruited = false(N, 0);
Bu.amp = zeros(N, 0);
Bu.delay = zeros(N, 0);
Bu.index = zeros(1, T);
Bu.mask = false(1, T);
Bu.level = zeros(1, T);
Bu.glow_ratio = zeros(1, T);
Bu.C_burst = zeros(N, T);

rs = RandStream('mt19937ar', 'Seed', seed);
% A [min max] range (or burst j's row of it), drawn uniformly. Inf stays Inf.
drng = @(r, j) draw(rs, r(min(j, size(r, 1)), :));
if isempty(p.onset)
    n = round(p.n);
    if n == 0
        return
    end
    D = arrayfun(@(j) round(drng(p.duration, j)), 1:n);
    gap = p.gap;
    lo = 1 + gap;
    assert(T - gap - lo + 1 >= sum(D) + (n - 1) * gap, 'Simulate_Ca_video:bursts', ...
        '%d bursts of %s frames, %d apart, do not fit in %d frames', n, mat2str(D), gap, T);
    for tries = 1:10000
        t0 = sort(lo + floor(rand(rs, 1, n) * (T - gap - lo + 1)));
        e = t0 + D - 1;
        if all(e <= T - gap) && (n == 1 || all(t0(2:end) - e(1:end-1) > gap)), break; end
    end
    assert(all(e <= T - gap) && (n == 1 || all(t0(2:end) - e(1:end-1) > gap)), ...
        'Simulate_Ca_video:bursts', 'could not space the bursts');
else
    t0 = sort(round(p.onset(:)'));
    n = numel(t0);
    D = arrayfun(@(j) round(drng(p.duration, j)), 1:n);
    assert(all(t0 >= 1) && all(t0 + D - 1 <= T), 'Simulate_Ca_video:bursts', 'burst outside the movie');
end

ss = @(x) (1 - cos(pi * min(max(x, 0), 1))) / 2;
t = 0:T-1;
peakC = max(C, [], 2);
for j = 1:n
    % Where, and who.
    cx = 1 + rand(rs) * (frame_size(2) - 1);
    cy = 1 + rand(rs) * (frame_size(1) - 1);
    R = drng(p.radius, j);
    speed = drng(p.speed, j);
    ratio = drng(p.neuropil, j);
    frac = drng(p.fraction, j);
    dist = hypot(xy(:, 1) - cx, xy(:, 2) - cy);
    rec = rand(rs, N, 1) < frac * exp(-dist.^2 / (2 * R^2));
    if ~any(rec)
        [~, m] = min(dist);
        rec(m) = true;                               % at least the nearest one
    end
    delay = dist / speed;                            % 0 when speed is Inf
    % The drive and the rhythmic discharges.
    u = (1:T) - t0(j) + 0.5;
    drive = min(ss(u / p.ramp), ss((D(j) - u) / p.ramp));
    drive(u < 0 | u > D(j)) = 0;
    ev = [];
    tt = t0(j) + rand(rs) * drng(p.interval, j);
    while tt < t0(j) + D(j)
        ev(end+1) = floor(tt); %#ok<AGROW>
        tt = tt + drng(p.interval, j);
    end
    ev = ev(ev >= 1 & ev <= T);
    amp = zeros(N, 1);
    pop = zeros(1, T);
    Cj = zeros(N, T);
    for i = find(rec)'
        q = drng(p.join, j);
        s = zeros(1, T);
        joined = rand(rs, 1, numel(ev)) < q * drive(ev);
        if ~any(joined)
            [~, m] = max(drive(ev));
            joined(m) = true;                        % every recruited neuron fires
        end
        ti = ev(joined) + round(delay(i));
        s(ti(ti <= T)) = 1;
        rise = 1 / lognrnd_rs(rs, p.rise(1), p.rise(2));
        decay = 1 / lognrnd_rs(rs, p.decay(1), p.decay(2));
        % The two draws overlap now and then; the slower one is the decay,
        % so the kernel stays positive (rise = decay would make it zero).
        [rise, decay] = deal(min(rise, decay), max(max(rise, decay), 1.01 * min(rise, decay)));
        amp(i) = drng(p.amplitude, j);
        if ~any(s)
            continue                                 % its wave arrives after the movie
        end
        g = exp(-t / decay) - exp(-t / rise);
        c = conv(s, g);
        c = c(1:T) / max(c);
        pop = pop + c;
        Cj(i, :) = amp(i) * peakC(i) * c;
    end
    pop = pop / max(pop);
    % The burst's frames: from its onset until the population trace has
    % decayed below p.tail of its peak; after that the burst adds nothing (a
    % later burst takes over its frames).
    [~, pk] = max(pop);
    last = find(pop >= p.tail, 1, 'last');
    pop(last+1:end) = 0;
    Cj(:, last+1:end) = 0;
    Bu.C_burst = Bu.C_burst + Cj;
    Bu.index(t0(j):last) = j;
    Bu.glow_ratio(t0(j):last) = ratio;
    Bu.level = Bu.level + pop;
    Bu.table(end+1, :) = [t0(j) D(j) sum(rec) sum(rec) / N last pk cx cy R speed ratio];
    Bu.recruited(:, j) = rec;
    Bu.amp(:, j) = amp .* rec;
    Bu.delay(:, j) = delay .* rec;
end
Bu.mask = Bu.index > 0;
end

function x = draw(rs, r)
% Uniform in [r(1) r(end)]; a range with Inf in it gives Inf.
if any(isinf(r))
    x = Inf;
else
    x = r(1) + rand(rs) * (r(end) - r(1));
end
end

function x = lognrnd_rs(rs, mu, sigma)
% lognrnd from a given stream.
x = exp(mu + sigma * randn(rs));
end
