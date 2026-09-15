function [ctrl_xy, cdx, cdy] = make_nonrigid_motion_ar(frame_size, T, grid_side, phi, std_px, max_px, seed)
%% Within-session NON-RIGID motion: a displacement field that varies over the frame.
%
% Translation moves every pixel by the same amount. Real tissue does not: it
% stretches, and parts of the field of view move differently from each other.
% Simulating only translation makes a piecewise-rigid corrector look bad no
% matter how good it is, because there is nothing for it to find and every patch
% shift it estimates is noise.
%
% Built the way PatchForge builds it, from two independent pieces:
%
%   TIME   each control point follows its own AR(1) walk, the same process
%          make_xy_motion_ar already uses for translation, so the deformation is
%          smooth from frame to frame rather than flickering.
%   SPACE  the control points sit on a regular grid over the frame, and a
%          thin-plate spline interpolates them into a dense per-pixel field, so
%          the deformation is smooth across the frame and has no patch seams.
%
% Independent walks per control point are what make the field non-rigid. Their
% mean is removed afterwards, so what is left is deformation only: any common
% component would be a translation, and translation is simulated separately.
%
% Inputs:
%   frame_size - [Ly Lx]
%   T          - number of frames
%   grid_side  - control points per axis. 3 gives a 3x3 grid, which is enough to
%                bend the frame without turning it into noise.
%   phi        - AR(1) coefficient, the temporal smoothness
%   std_px     - stationary standard deviation of each control point, in pixels
%   max_px     - optional cap on the peak displacement
%   seed       - random seed
%
% Outputs:
%   ctrl_xy - [n 2] control point positions, (x,y), in pixels
%   cdx,cdy - [n T] displacement of each control point at each frame
%
% Author: Pablo Vergara

if nargin<3 || isempty(grid_side), grid_side = 3; end
if nargin<4 || isempty(phi),       phi = 0.95;    end
if nargin<5 || isempty(std_px),    std_px = 2;    end
if nargin<6,                       max_px = [];   end
if nargin>=7 && ~isempty(seed),    rng(seed);     end

Ly = frame_size(1); Lx = frame_size(2);
gx = linspace(1, Lx, grid_side);
gy = linspace(1, Ly, grid_side);
[GX, GY] = meshgrid(gx, gy);
ctrl_xy = [GX(:), GY(:)];
n = size(ctrl_xy,1);

sigma = std_px*sqrt(1-phi^2);
cdx = filter(1,[1 -phi], sigma*randn(n,T), [], 2);
cdy = filter(1,[1 -phi], sigma*randn(n,T), [], 2);

% Each control point centred over time, so no point carries a standing offset.
cdx = cdx - mean(cdx,2);
cdy = cdy - mean(cdy,2);

% And the frame centred at each instant: a displacement shared by every control
% point is a translation, which is simulated separately. What is left is only
% the part that varies ACROSS the frame, which is what non-rigid means.
cdx = cdx - mean(cdx,1);
cdy = cdy - mean(cdy,1);

if ~isempty(max_px) && max_px > 0
    sc = max(1, max(abs([cdx(:); cdy(:)]))/max_px);
    cdx = cdx./sc; cdy = cdy./sc;
end
end
