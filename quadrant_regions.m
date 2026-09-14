function [masks, region_id, info] = quadrant_regions(d1, d2, margin, side)
%QUADRANT_REGIONS  Square placement window split into 2x2 quadrants.
%
%   [masks, region_id, info] = quadrant_regions(d1, d2, margin, side)
%
%   Neurons are placed ONLY inside a centred square window, never within
%   `margin` pixels of the frame edge. Everything outside the window gets no
%   neurons, which keeps the padding/edge artifacts of the motion-correction
%   crop and the CNMF patch border away from every component being measured.
%
%   The window is divided into four equal quadrants so that spatial density
%   and temporal correlation can be crossed as a balanced 2x2 factorial:
%
%       masks{1}  top-left       masks{2}  top-right
%       masks{3}  bottom-left    masks{4}  bottom-right
%
%   A crossed design is the point: the previous layout grew four irregular
%   regions with density and correlation confounded, so the low-density /
%   high-correlation cell never materialised and the two factors could not be
%   separated.
%
%   INPUTS
%     d1, d2   frame size in pixels (the SAVED frame, after the simulator's
%              20 px crop per side)
%     margin   pixels of forbidden border around the window
%     side     optional; window side in pixels. Default is the largest square
%              that fits inside the margins.
%
%   OUTPUTS
%     masks      1x4 cell of logical d1 x d2 masks, mutually exclusive
%     region_id  d1 x d2 int, 0 outside the window and 1..4 inside
%     info       struct with the window geometry and quadrant area

if nargin < 3 || isempty(margin), margin = 20; end
maxside = min(d1, d2) - 2 * margin;
if nargin < 4 || isempty(side), side = maxside; end
if side > maxside
    error('quadrant_regions:tooBig', ...
        'side %d exceeds what fits inside the margins (%d).', side, maxside);
end
side = 2 * floor(side / 2);            % even, so quadrants are equal

r0 = floor((d1 - side) / 2);
c0 = floor((d2 - side) / 2);
half = side / 2;

region_id = zeros(d1, d2, 'uint8');
masks = cell(1, 4);
q = 0;
for br = 0:1
    for bc = 0:1
        q = q + 1;
        m = false(d1, d2);
        m(r0 + br*half + (1:half), c0 + bc*half + (1:half)) = true;
        masks{q} = m;
        region_id(m) = q;
    end
end

info = struct('row0', r0, 'col0', c0, 'side', side, 'half', half, ...
              'quad_area', half^2, 'window_area', side^2, ...
              'frame_area', d1*d2, 'occupancy', side^2/(d1*d2));
end
