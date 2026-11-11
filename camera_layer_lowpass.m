function L = camera_layer_lowpass(F, o)
%% The out-of-focus light of a still frame: a low-pass of it (make_camera_optics).
%
% o.halfpoint h is the scale of the split, as the sigma of the Gaussian whose
% transfer is 1/2 at the same spatial frequency; the shapes differ in how
% sharp the transition is:
%   'gaussian'  imgaussfilt(F, h)
%   'dog'       2 G(a) - G(sqrt(2) a), a = 1.331 h: a difference of Gaussians
%               that keeps the coarse scales whole and drops faster (its
%               transfer 2x - x^2, x the Gaussian's, is 1/2 at x = 0.293)
%   'opening'   a grey-scale opening with a disk of radius round(h), the
%               background estimate of MIN1PIPE-style pipelines (bright
%               structures narrower than the disk go to the in-focus part),
%               smoothed by a Gaussian of h/4
% With o.edge_factor e > 1 the blur grows toward the frame edge: the low-pass
% at h and at e*h, blended by o.edge_weight (0 centre, 1 corners).
%
% Author: Pablo Vergara

L = lowpass(F, o.shape, o.halfpoint);
if o.edge_factor > 1
    L2 = lowpass(F, o.shape, o.halfpoint * o.edge_factor);
    L = (1 - o.edge_weight) .* L + o.edge_weight .* L2;
end
end

function L = lowpass(F, shape, h)
switch shape
    case 'gaussian'
        L = imgaussfilt(F, h);
    case 'dog'
        a = 1.331 * h;
        L = 2 * imgaussfilt(F, a) - imgaussfilt(F, sqrt(2) * a);
    case 'opening'
        L = imopen(F, strel('disk', max(1, round(h))));
        L = imgaussfilt(L, max(0.5, h / 4));
    otherwise
        error('camera_layer_lowpass:shape', 'unknown out-of-focus filter "%s"', shape);
end
end
