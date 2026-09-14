function [masks, region_id, info] = grid_regions(d1, d2, margin, nr, nc, side)
%GRID_REGIONS  Square placement window split into an nr x nc grid of cells.
%
%   [masks, region_id, info] = grid_regions(d1, d2, margin, nr, nc, side)
%
%   Generalises quadrant_regions from 2x2 to any grid, so density and temporal
%   correlation can be varied over a RANGE rather than at two levels each. With
%   only two levels of each factor the design has four distinct conditions,
%   which is enough to test whether a factor matters but not enough to fit a
%   rule as a function of it.
%
%   Cells are numbered row by row: cell (i,j) is (i-1)*nc + j.
%
%   INPUTS
%     d1, d2   frame size in pixels, on the SAVED frame
%     margin   forbidden border around the window, in pixels
%     nr, nc   grid size, in cells
%     side     optional window side in pixels; default is the largest square
%              that fits inside the margins and divides evenly into the grid
%
%   Author: Pablo Vergara

if nargin < 3 || isempty(margin), margin = 20; end
if nargin < 4 || isempty(nr), nr = 4; end
if nargin < 5 || isempty(nc), nc = nr; end
maxside = min(d1, d2) - 2 * margin;
if nargin < 6 || isempty(side), side = maxside; end
if side > maxside
    error('grid_regions:tooBig', ...
        'side %d exceeds what fits inside the margins (%d).', side, maxside);
end
step = floor(side / max(nr, nc));      % square cells, equal area
side = step * max(nr, nc);

r0 = floor((d1 - step*nr) / 2);
c0 = floor((d2 - step*nc) / 2);
region_id = zeros(d1, d2, 'uint8');
masks = cell(1, nr*nc);
for i = 1:nr
    for j = 1:nc
        q = (i-1)*nc + j;
        m = false(d1, d2);
        m(r0 + (i-1)*step + (1:step), c0 + (j-1)*step + (1:step)) = true;
        masks{q} = m;
        region_id(m) = q;
    end
end
info = struct('row0', r0, 'col0', c0, 'step', step, 'nr', nr, 'nc', nc, ...
              'cell_area', step^2, 'window_area', (step*nr)*(step*nc), ...
              'frame_area', d1*d2);
end
