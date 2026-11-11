function G = make_glitches(frame_size, T, p, seed)
%% Acquisition glitches: frames the sensor or the link spoils, in short runs.
%
% Real one-photon recordings have runs of 1-8 spoiled frames, after which the
% next frame is normal again. They are acquisition, not motion: they happen to
% the recorded frame, whatever the tissue does. Four kinds:
%
%   1 dropped  the frame never arrived; the writer repeats the last frame it
%              delivered (frozen)
%   2 white    the frame is flooded: g * frame + o, saturating, with gain g
%              (log-uniform in white_gain) and offset o (white_offset, grey
%              levels); the dark periphery may stay partly visible
%   3 black    the frame is (nearly) black: g * frame + o with g in
%              [0 black_residual] and o in black_level
%   4 partial  part of the frame is spoiled, along whole rows as the sensor
%              reads them out: a cap from the top or the bottom edge (with
%              probability partial_cap), else a band in between, covering a
%              fraction partial_band of the frame's height; the spoiled rows
%              are white (probability partial_white) or black, as above
%
% Each run has one kind, drawn from the weights p.kinds; with probability
% p.mixed a run draws a kind for every frame instead. apply_glitches applies
% them to the recorded uint8 frames. Everything is drawn here, from a stream of
% its own, so the rest of the simulation is the same with or without glitches,
% and the same glitches land on a run with and without noise.
%
% Inputs:
%   frame_size - [Ly Lx]
%   T          - number of frames
%   p          - struct, the glitch_* options without the prefix (see
%                Simulate_Ca_video): rate (runs per 1000 frames), run, kinds,
%                mixed, white_gain, white_offset, black_residual, black_level,
%                partial_band, partial_cap, partial_white, inset, gap
%   seed       - seed of the stream
%
% Output, struct G:
%   names         the kind names, 1..4
%   runs [n 3]    one row per run: [t0 length kind] (kind 0 = mixed)
%   kind [1 T]    0 = normal frame, else 1..4
%   mask [1 T]    kind > 0
%   gain, offset [1 T]  the transform of white, black and partial frames
%                 (1 and 0 elsewhere)
%   rows [2 T]    first and last spoiled row (1-based; the whole frame for
%                 white and black, 0 0 for normal and dropped frames)
%
% Author: Pablo Vergara

Ly = frame_size(1);
G = struct();
G.names = {'dropped', 'white', 'black', 'partial'};
G.runs = zeros(0, 3);
G.kind = zeros(1, T);
G.mask = false(1, T);
G.gain = ones(1, T);
G.offset = zeros(1, T);
G.rows = zeros(2, T);

rs = RandStream('mt19937ar', 'Seed', seed);
n = round(p.rate * T / 1000);
if n == 0
    return
end
len = p.run(1) + floor(rand(rs, 1, n) * (p.run(end) - p.run(1) + 1));
gap = max(1, p.gap);
% Place the runs: never on the first frame (a dropped frame repeats the one
% before), at least gap normal frames between runs.
need = sum(len) + n * gap;
assert(need <= T - 1, 'Simulate_Ca_video:glitches', '%d glitch runs do not fit in %d frames', n, T);
% Free frames to share out, then a random composition of them.
free = T - 1 - need;
cut = sort(floor(rand(rs, 1, n) * (free + 1)));
pre = diff([0, cut]);                            % free frames before each run
t = 2;
w = p.kinds(:)' / sum(p.kinds);
for j = 1:n
    t = t + pre(j);
    mixed = rand(rs) < p.mixed;
    k_run = draw_kind(rs, w);
    G.runs(end+1, :) = [t len(j) k_run * ~mixed];
    for k = t:t + len(j) - 1
        kind = k_run;
        if mixed, kind = draw_kind(rs, w); end
        G.kind(k) = kind;
        switch kind
            case 2
                [G.gain(k), G.offset(k)] = white(rs, p);
                G.rows(:, k) = [1; Ly];
            case 3
                [G.gain(k), G.offset(k)] = black(rs, p);
                G.rows(:, k) = [1; Ly];
            case 4
                if rand(rs) < p.partial_white
                    [G.gain(k), G.offset(k)] = white(rs, p);
                else
                    [G.gain(k), G.offset(k)] = black(rs, p);
                end
                H = Ly - 2 * p.inset;
                h = max(1, round((p.partial_band(1) + rand(rs) * diff(p.partial_band([1 end]))) * H));
                if rand(rs) < p.partial_cap
                    if rand(rs) < 0.5
                        G.rows(:, k) = [1; p.inset + h];           % cap from the top
                    else
                        G.rows(:, k) = [Ly - p.inset - h + 1; Ly]; % from the bottom
                    end
                else
                    a = p.inset + 1 + floor(rand(rs) * (H - h + 1));
                    G.rows(:, k) = [a; a + h - 1];                 % band
                end
        end
    end
    t = t + len(j) + gap;
end
G.mask = G.kind > 0;
end

function k = draw_kind(rs, w)
k = find(rand(rs) < cumsum(w), 1);
end

function [g, o] = white(rs, p)
g = exp(log(p.white_gain(1)) + rand(rs) * diff(log(p.white_gain([1 end]))));
o = p.white_offset(1) + rand(rs) * diff(p.white_offset([1 end]));
end

function [g, o] = black(rs, p)
g = rand(rs) * p.black_residual;
o = p.black_level(1) + rand(rs) * diff(p.black_level([1 end]));
end
