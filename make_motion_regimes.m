function R = make_motion_regimes(frame_size, T, p, seed)
%% Within-session motion that changes CHARACTER over time.
%
% The AR(1) shake and the 3x3 spline bend are stationary: every stretch of the
% recording looks like every other. Real recordings are not like that. The
% tissue sits still apart from a rigid shake for a while, then bends; a local
% patch swells or slides on its own and settles again; and now and then the
% whole field of view jumps when the animal knocks the scope. This draws those
% three things, as plain numbers, so they can be applied to every frame and
% written out as ground truth.
%
%   REGIMES       alternating rigid-only and bend stretches. bend_weight(t) is
%                 0 in a rigid stretch and 1 in a bend stretch, with a smooth
%                 ramp at each change; it multiplies every non-rigid part (the
%                 spline bend and the local bends), never the rigid shake.
%   LOCAL BENDS   Gaussian bumps of displacement at random places, of several
%                 widths and amplitudes, each fading in and out over its own
%                 lifetime. 'shift' bumps slide a patch of tissue in one
%                 direction; 'swell' bumps push it out from (or pull it into) a
%                 centre, so a patch that contains one moves by almost nothing
%                 as a whole and only finer patches can see it.
%   JUMPS         sudden rigid steps of the whole field of view. A held jump
%                 moves the resting position for good; a decaying one returns
%                 to it exponentially. Each jump lands within jump_bound of
%                 where the recording started.
%
% Everything is drawn from a stream of its own, so the rest of the simulation
% (neurons, spikes, noise) is the same with or without it.
%
% Inputs:
%   frame_size - [Ly Lx] of the frame the motion is applied to
%   T          - number of frames
%   p          - struct of options, the within_session_* names without the
%                prefix (see Simulate_Ca_video): rigid_frames, bend_frames,
%                regime_ramp, local_bends, local_bend_width, local_bend_amp,
%                local_bend_life, local_bend_strain, local_bend_inset, jumps,
%                jump_size, jump_held, jump_decay, jump_bound, jump_gap
%   seed       - seed of the stream
%
% Output, struct R:
%   bend_weight [1 T]  weight of the non-rigid motion at each frame, in [0 1]
%   stretches   [k 3]  [first last is_bend] of each regime stretch, frames
%   bumps       [m 10] one row per local bend: [cx cy sigma amp angle kind
%                      t_on t_off sign peak_strain]; (cx, cy) in pixels of the
%                      frame (1-based x = column, y = row), sigma and amp in
%                      pixels, angle in radians (shift bumps), kind 1 = shift,
%                      2 = swell, sign +1 = outward swell, -1 = inward.
%   jumps       [j 5]  one row per jump: [t0 dx dy held tau]. The step happens
%                      between frames t0-1 and t0; (dx, dy) in pixels; held 1
%                      or 0; tau the decay time constant in frames, Inf if held.
%   jump_cols         names of the jump columns
%   jump_xy     [2 T]  rigid offset from the jumps at each frame, pixels
%   bump_cols         names of the bump columns
%
% Field of a bump at pixel (x, y) and frame t in [t_on, t_off]:
%   e(t)  = sin(pi*(t - t_on + 1)/(t_off - t_on + 2))^2       lifetime envelope
%   g     = exp(-((x-cx)^2 + (y-cy)^2) / (2*sigma^2))
%   shift: amp*e(t)*g*[cos(angle), sin(angle)]
%   swell: sign*amp*e(t)*g*sqrt(exp(1))*[(x-cx)/sigma, (y-cy)/sigma]
% and amp is the peak displacement of either kind.
%
% Author: Pablo Vergara

rs = RandStream('mt19937ar', 'Seed', seed);
Ly = frame_size(1); Lx = frame_size(2);

R = struct();
R.bump_cols = {'cx', 'cy', 'sigma', 'amp', 'angle', 'kind', 't_on', 't_off', 'sign', 'peak_strain'};
R.jump_cols = {'t0', 'dx', 'dy', 'held', 'tau'};

%% Regimes
if isempty(p.rigid_frames)
    R.stretches = [1 T 1];
    R.bend_weight = ones(1, T);
else
    R.stretches = zeros(0, 3);
    is_bend = rand(rs) < 0.5;
    t = 1;
    while t <= T
        lim = p.rigid_frames;
        if is_bend, lim = p.bend_frames; end
        len = lim(1) + floor(rand(rs) * (lim(end) - lim(1) + 1));
        last = min(T, t + len - 1);
        R.stretches(end+1, :) = [t last is_bend];
        t = last + 1;
        is_bend = ~is_bend;
    end
    R.bend_weight = zeros(1, T);
    ramp = p.regime_ramp;
    for k = find(R.stretches(:, 3))'
        a = R.stretches(k, 1); b = R.stretches(k, 2);
        tt = a:b;
        r = ones(size(tt));
        if a > 1, r = min(r, (tt - a + 1) / (ramp + 1)); end
        if b < T, r = min(r, (b - tt + 1) / (ramp + 1)); end
        R.bend_weight(tt) = sin(pi/2 * r).^2;
    end
end

%% Local bends, drawn inside the bend stretches only
R.bumps = zeros(0, numel(R.bump_cols));
if p.local_bends > 0
    life = p.local_bend_life;
    mean_life = mean(life([1 end]));
    rate = p.local_bends / mean_life;                % onsets per frame
    inset = p.local_bend_inset;
    for k = find(R.stretches(:, 3))'
        a = R.stretches(k, 1); b = R.stretches(k, 2);
        % Start the process one lifetime early so the stretch opens with about
        % local_bends of them already under way (they fade in with the stretch).
        t = a - life(end) - log(rand(rs)) / rate;
        while t <= b
            on = floor(t);
            len = life(1) + floor(rand(rs) * (life(end) - life(1) + 1));
            t_on = max(on, a); t_off = min(on + len - 1, b);
            sigma = exp(log(p.local_bend_width(1)) + rand(rs) * diff(log(p.local_bend_width([1 end]))));
            amp = p.local_bend_amp(1) + rand(rs) * diff(p.local_bend_amp([1 end]));
            angle = 2*pi*rand(rs);
            kind = 1 + (rand(rs) < 0.5);
            sgn = 1 - 2*(rand(rs) < 0.5);
            cx = inset + 1 + rand(rs) * (Lx - 2*inset - 1);
            cy = inset + 1 + rand(rs) * (Ly - 2*inset - 1);
            % Peak strain (largest gradient of the displacement) of a unit bump:
            % 1/(sigma*sqrt(e)) for a shift, sqrt(e)/sigma for a swell (at its
            % centre). Capping it keeps the tissue from folding over.
            g1 = [exp(-0.5), sqrt(exp(1))] / sigma;
            amp = min(amp, p.local_bend_strain / g1(kind));
            if t_off >= t_on
                R.bumps(end+1, :) = [cx cy sigma amp angle kind t_on t_off sgn amp*g1(kind)];
            end
            t = t - log(rand(rs)) / rate;
        end
    end
end

%% Jumps
R.jumps = zeros(0, numel(R.jump_cols));
R.jump_xy = zeros(2, T);
n_jump = round(p.jumps * T / 1000);
if n_jump > 0
    gap = p.jump_gap;
    lo = gap; hi = T - 1;                            % no jump on the first frames
    assert(hi - lo >= (n_jump - 1) * gap, ...
        'Simulate_Ca_video:jumps', '%d jumps %d frames apart do not fit in %d frames', n_jump, gap, T);
    for tries = 1:10000
        t0 = sort(lo + floor(rand(rs, 1, n_jump) * (hi - lo + 1)));
        if n_jump == 1 || all(diff(t0) >= gap), break; end
    end
    assert(n_jump == 1 || all(diff(t0) >= gap), 'Simulate_Ca_video:jumps', 'could not space the jumps');
    rest = [0; 0];                                   % resting position
    for j = 1:n_jump
        sz = p.jump_size(1) + rand(rs) * diff(p.jump_size([1 end]));
        held = rand(rs) < p.jump_held;
        for tries = 1:10000
            ang = 2*pi*rand(rs);
            J = sz * [cos(ang); sin(ang)];
            if norm(rest + J) <= p.jump_bound, break; end
        end
        assert(norm(rest + J) <= p.jump_bound + 1e-9, 'Simulate_Ca_video:jumps', 'jump_bound too tight');
        tt = t0(j):T;
        if held
            tau = Inf;
            R.jump_xy(:, tt) = R.jump_xy(:, tt) + J;
            rest = rest + J;
        else
            tau = p.jump_decay(1) + rand(rs) * diff(p.jump_decay([1 end]));
            R.jump_xy(:, tt) = R.jump_xy(:, tt) + J * exp(-(tt - t0(j)) / tau);
        end
        R.jumps(end+1, :) = [t0(j) J' held tau];
    end
end
end
