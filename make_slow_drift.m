function Dr = make_slow_drift(frame_size, T, p, seed)
%% Slow drift: a session-long deformation of the tissue that builds up at a rate.
%
% Real sessions drift slowly (tissue settling, vessels dilating, the focal
% plane moving). The drift is mostly a rigid translation, with a small affine
% part (scale, rotation, shear) and small broad bends, and over a session it
% does not average to zero. This draws it as plain numbers, per frame, so it can
% be applied to every frame and written out as ground truth.
%
% Each part follows a slow random path set by a RATE, not by a final size, so a
% longer session drifts further. Time is the drift's own clock in minutes:
% minutes per movie = p.minutes, or the movie's own length at 25 frames per
% second (the AVI writer's rate, as impact_duration uses) when empty.
%
%   rigid      r(t) = rate * integral of [cos phi(t'), sin phi(t')] dt'
%   scale s, rotation theta, shear e (each its own path)
%              x(t) = rate_x * integral of cos phi_x(t') dt'
%   bend k     a_k(t) = rate_k * integral of cos phi_k(t') dt'
%
% phi is a heading that wanders: a random walk over knots p.knot minutes
% apart (step sd = wander * sqrt(knot), wander in rad per sqrt(minute)),
% interpolated by a cubic spline, so the path is smooth and turns slowly.
% A heading that turns by more than pi/2 makes a 1-D part (and a component of
% the rigid path) go back; the net displacement is still not zero.
% Every part is 0 at the first frame: the first frame is the reference
% position (the motion-free twin). The middle frame's value is what a
% middle-window reference sees.
%
% The field at pixel p = (x, y) (1-based, x = column, y = row), frame t:
%
%   D(p, t) = r(t) + G(t) (p - c) + sum_k a_k(t) g_k(p) [cos ang_k, sin ang_k]
%   G = [s + e cos 2al,  e sin 2al - theta;  e sin 2al + theta,  s - e cos 2al]
%   g_k(p) = exp(-|p - c_k|^2 / (2 sigma_k^2))
%
% c is the frame centre, al the shear axis. slow_drift_field evaluates it;
% compose_slow_drift composes it with the rest of the within-session motion.
% D is the distance the tissue moved, as every field here: the frame shows at
% p what the still tissue had at p - D(p).
%
% Everything is drawn from a stream of its own, so the rest of the simulation
% is the same with or without it.
%
% Inputs:
%   frame_size - [Ly Lx] of the frame the motion is applied to
%   T          - number of frames
%   p          - struct, the slow_drift_* options without the prefix (see
%                Simulate_Ca_video): minutes, rate, wander, scale, rotation,
%                shear, bends, bend_width, bend_rate, knot
%   seed       - seed of the stream
%
% Output, struct Dr:
%   minutes, t_min [1 T]  the drift's clock: minutes since the first frame
%   centre [1 2]          c, (x, y)
%   r      [2 T]          rigid part, px
%   G      [4 T]          [gxx; gxy; gyx; gyy], the affine part's gradient
%   scale, rotation, shear [1 T]   s (fraction), theta (rad), e (fraction)
%   shear_axis            al (rad)
%   bends  [K 4]          [cx cy sigma angle] per bend (px, px, px, rad)
%   bend_amp [K T]        a_k(t), px (signed, along angle)
%   rates                 the drawn rates: rigid px/min, scale and shear
%                         fraction/min, rotation rad/min, bends px/min [1 K]
%   wander                the drawn wander, rad/sqrt(min)
%   mid_frame             the middle frame, floor(T/2) + 1
%
% Author: Pablo Vergara

rs = RandStream('mt19937ar', 'Seed', seed);
Ly = frame_size(1); Lx = frame_size(2);
if isempty(p.minutes)
    M = T / (25 * 60);
else
    M = p.minutes;
end
dt = M / T;
Dr = struct();
Dr.minutes = M;
Dr.t_min = (0:T-1) * dt;
Dr.centre = [(Lx + 1) / 2, (Ly + 1) / 2];
Dr.mid_frame = floor(T / 2) + 1;

w = uniform(rs, p.wander);
Dr.wander = w;
knots = 0:p.knot:(M + p.knot);

% Rigid
rate = log_uniform(rs, p.rate);
phi = heading(rs, knots, Dr.t_min, w);
Dr.r = integrate(rate * [cos(phi); sin(phi)], dt);
Dr.rates.rigid = rate;

% Affine: scale (%/min), rotation (deg/min), shear (%/min, about a random axis)
rate_s = log_uniform(rs, p.scale) / 100;
Dr.scale = integrate(rate_s * cos(heading(rs, knots, Dr.t_min, w)), dt);
rate_r = log_uniform(rs, p.rotation) * pi / 180;
Dr.rotation = integrate(rate_r * cos(heading(rs, knots, Dr.t_min, w)), dt);
rate_e = log_uniform(rs, p.shear) / 100;
Dr.shear_axis = pi * rand(rs);
Dr.shear = integrate(rate_e * cos(heading(rs, knots, Dr.t_min, w)), dt);
Dr.rates.scale = rate_s; Dr.rates.rotation = rate_r; Dr.rates.shear = rate_e;
c2 = cos(2 * Dr.shear_axis); s2 = sin(2 * Dr.shear_axis);
Dr.G = [Dr.scale + Dr.shear * c2; Dr.shear * s2 - Dr.rotation; ...
        Dr.shear * s2 + Dr.rotation; Dr.scale - Dr.shear * c2];

% Broad bends: centres in the middle 70 % of the frame
K = round(p.bends);
Dr.bends = zeros(K, 4);
Dr.bend_amp = zeros(K, T);
Dr.rates.bends = zeros(1, K);
for k = 1:K
    cx = 1 + (0.15 + 0.7 * rand(rs)) * (Lx - 1);
    cy = 1 + (0.15 + 0.7 * rand(rs)) * (Ly - 1);
    sigma = log_uniform(rs, p.bend_width);
    ang = 2 * pi * rand(rs);
    rate_k = log_uniform(rs, p.bend_rate);
    Dr.bends(k, :) = [cx cy sigma ang];
    Dr.bend_amp(k, :) = integrate(rate_k * cos(heading(rs, knots, Dr.t_min, w)), dt);
    Dr.rates.bends(k) = rate_k;
end
end

function phi = heading(rs, knots, t, w)
% A heading that wanders: random walk on the knots, cubic spline between.
n = numel(knots);
steps = w * sqrt(knots(2) - knots(1)) * randn(rs, 1, n - 1);
pk = 2 * pi * rand(rs) + [0, cumsum(steps)];
phi = interp1(knots, pk, t, 'spline');
end

function x = integrate(v, dt)
% Left Riemann sum, 0 at the first frame. v [m T] per minute.
x = [zeros(size(v, 1), 1), cumsum(v(:, 1:end-1) * dt, 2)];
end

function x = uniform(rs, lim)
x = lim(1) + rand(rs) * (lim(end) - lim(1));
end

function x = log_uniform(rs, lim)
if lim(1) == lim(end)
    x = lim(1);
    rand(rs);                     % keep the stream's position
    return
end
x = exp(log(lim(1)) + rand(rs) * (log(lim(end)) - log(lim(1))));
end
