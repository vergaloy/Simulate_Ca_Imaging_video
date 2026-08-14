function [Aout,bA,C,S,LFP,placement]=create_neuron_data(opt)
%% Create Mask
if opt.create_mask
    BW=create_heart_mask(opt.d);
else
    BW=ones(opt.d);
end
BW = apply_placement_margin(BW, opt);
%% Create random spatial
[Aout,bA,placement]=create_spatial(BW,opt);
N=size(Aout{1,1},3);
%% Adjust spike probability and kinetics to the sf
spike_prob=opt.spike_prob;
spike_prob(1)=log(exp(spike_prob(1))/opt.sf);
spike_prob(2)=spike_prob(2)/opt.sf;
invtRise=opt.invtRise;
invtRise(1)=log(exp(invtRise(1))/opt.sf);
invtDecay=opt.invtDecay;
invtDecay(1)=log(exp(invtDecay(1))/opt.sf);

%% create spike probability distribution
P=lognrnd(spike_prob(1),spike_prob(2),size(Aout{1, 1},3),1);
C=[];
LFP=[];
S=[];
if has_region_layout(opt)
    % Spatially coherent activity: neurons that sit close together fire
    % together, with a strength given by the region each neuron belongs to.
    % correlated_spikes replaces the independent Bernoulli draw of
    % create_random_spike; everything downstream is unchanged.
    if opt.LFP > 0
        warning('create_neuron_data:lfpIgnored', ...
            ['LFP modulation is not applied when region_areas is used; ' ...
             'the spike times come from correlated_spikes instead.']);
    end
    cs = placement.corr_strength(:);
    for i=1:opt.ses
        s = correlated_spikes(placement.X, placement.Y, P, opt.F, cs, ...
            opt.corr_length, placement.gSig_fullres);
        if opt.force_active==1
            s = force_active_rows(s);
        end
        LFP=[LFP,ones(1,opt.F)];
        S=[S,s];
    end
else
    for i=1:opt.ses
        [s,lfp]=create_random_spike(N,opt.F,opt.sf,P,opt.LFP,1,opt.force_active);
        LFP=[LFP,lfp];
        S=[S,s];
    end
end
S=fix_inactive(S);
C=conv_spikes(S,invtRise,invtDecay,size(S,2));


% C=C.*(1+randn(length(X),1).*0.2);
%

end


function [Aout,bA,placement]=create_spatial(BW,opt)
if has_region_layout(opt)
    % Random continuous regions grown on the SAVED-frame geometry, each with
    % its own neuron count, minimum separation and activity coherence.
    [X,Y,placement]=place_regions(BW,opt);
elseif has_dense_region(opt)
    % Spatially heterogeneous placement: one or more dense rectangles plus a
    % sparse background. See place_heterogeneous below.
    [X,Y,placement]=place_heterogeneous(BW,opt);
else
    % Original, homogeneous placement. Untouched so that a call without the
    % dense_region arguments reproduces earlier simulations bit for bit.
    [y, x] = find(BW);
    [X,Y]=get_random_points(x,y,opt.min_dist,opt.Nneu,opt.plotme);
    placement = struct('X',X,'Y',Y,'X_saved',X-20,'Y_saved',Y-20, ...
        'region_id',ones(1,numel(X)),'dense_region_full',[], ...
        'dense_region_saved',[],'n_per_region',numel(X), ...
        'min_dist_per_region',opt.min_dist);
end
if opt.CA1_A
    load('A_CA1.mat');
elseif opt.A2
    load('A2.mat');
else
    load('A.mat');
end
N=size(X,2);

if size(A,4)>1
    ix = randperm(size(A,4),N);
    w=linspace(0,1,opt.ses);
    for i=1:length(w)
        As(:,:,:,i)=squeeze(A(:,:,1,ix))*(1-w(i))+squeeze(A(:,:,2,ix))*(w(i));
    end
else
    ix = datasample(1:size(A,3),N);
    As=repelem(A(:,:,ix),1,1,1,opt.ses);
end

if opt.scale>1
    [d1, d2, s, c] = size(As);
    temp = zeros(round(d1*opt.scale), round(d2*opt.scale), s, c, 'like', As);

    for si = 1:s
        for ci = 1:c
            temp(:,:,si,ci) = imresize(As(:,:,si,ci), opt.scale, 'bicubic');
        end
    end
    As=temp;
    clear temp
end

% Effective neuron radius of the footprints that were actually drawn, in
% full-resolution pixels. This is the gSig that correlated_spikes needs in
% order to turn a correlation length expressed in gSig units into pixels.
% Measuring it from As rather than hard-coding it keeps it correct when
% opt.scale resizes the kernels or a different footprint library is used.
placement.gSig_fullres = footprint_gsig(As(:,:,:,1));

for s=1:opt.ses
    Aout_t=[];
    bA_t=[];
    for i=1:length(X)
        Z=zeros(opt.d);
        Z(Y(i),X(i))=1;
        Aout_t(:,:,i)=conv2(Z, As(:,:,i,s), 'same');
        bA_t(:,:,i)=imgaussfilt(Aout_t(:,:,i),10,'FilterSize',opt.d(2)+1);
    end
    Aout_t=Aout_t./max(Aout_t,[],'all');
    bA_t=bA_t./max(bA_t,[],'all');
    if opt.plotme
        figure;imshow(max(Aout_t,[],3));
        figure;imshow(max(bA_t,[],3));
    end
    Aout{s}=Aout_t;
    bA{s}=bA_t;
end

end


%% ---------------------------------------------------------------------
%  Spatially heterogeneous neuron placement
%  ---------------------------------------------------------------------
function BW = apply_placement_margin(BW, opt)
% Optionally forbid neuron centres within opt.placement_margin pixels of the
% FULL-frame border. Off by default ([]), in which case BW is returned
% untouched and placement is bit-for-bit what it always was.
%
% This matters only when the heart mask is disabled (create_mask=false): the
% saved video drops 20 px per border and the neuron kernels are 50 x 50, so a
% neuron centred closer than ~45 px to the border ends up with a ground-truth
% footprint that is clipped or empty in the saved frame. With the heart mask on,
% placement is already far from the border and this option changes nothing.
if ~isfield(opt,'placement_margin') || isempty(opt.placement_margin)
    return
end
m = round(opt.placement_margin);
if m <= 0
    return
end
d1 = size(BW,1); d2 = size(BW,2);
if 2*m >= min(d1,d2)
    error('create_neuron_data:placementMargin', ...
        'placement_margin (%d) leaves no room in a %d x %d frame.', m, d1, d2);
end
keep = false(d1,d2);
keep(m+1:d1-m, m+1:d2-m) = true;
BW = BW & keep;
end


function tf = has_dense_region(opt)
% True when the caller asked for a heterogeneous field of view.
tf = isfield(opt,'dense_region') && ~isempty(opt.dense_region);
end


%% ---------------------------------------------------------------------
%  Random continuous regions with per-region density and coherence
%  ---------------------------------------------------------------------
function tf = has_region_layout(opt)
% True when the caller asked for the region-based layout (region_areas).
tf = isfield(opt,'region_areas') && ~isempty(opt.region_areas);
end


function [X,Y,place] = place_regions(BW,opt)
% Partition the field of view into R random continuous regions and place
% neurons in them, one density and one activity coherence per region.
%
% The regions are grown on the SAVED-frame geometry, i.e. on the frame that is
% actually written to disk. Add_NRmotion drops 20 px on every border, so the
% saved frame is opt.d - 40 on each axis. opt.region_areas is therefore read in
% saved-frame pixels, exactly like the 'saved' convention of dense_region, and
% the masks are shifted by 20 px when they are lifted into the full frame that
% the simulation itself runs on. Neurons then never land in the strip that is
% cropped away, so every neuron has a complete footprint in the saved video.

crop = 20;
d1 = size(BW,1);
d2 = size(BW,2);
s1 = d1 - 2*crop;                       % saved-frame height
s2 = d2 - 2*crop;                       % saved-frame width
if s1 < 1 || s2 < 1
    error('create_neuron_data:regionFrame', ...
        'The frame (%d x %d) is too small to leave a saved frame after the 20 px crop.', d1, d2);
end

areas = double(opt.region_areas(:))';
R = numel(areas);

% --- region layout ---------------------------------------------------------
if isfield(opt,'region_seed') && ~isempty(opt.region_seed)
    layout_seed = opt.region_seed;
else
    layout_seed = opt.seed;             % fall back to the main simulation seed
end
% grow_random_regions restores the global random state on exit, so the layout
% seed does not disturb the rest of the simulation.
[masks_saved, region_id_image, areas_saved] = ...
    grow_random_regions(s1, s2, areas, layout_seed);

% Lift the masks into the full frame and intersect with the (heart) mask.
region_masks = cell(1, R);
for r = 1:R
    m = false(d1, d2);
    m(crop+1:crop+s1, crop+1:crop+s2) = masks_saved{r};
    region_masks{r} = m & logical(BW);
end
areas_allowed = cellfun(@(m) sum(m(:)), region_masks);
if any(areas_allowed == 0)
    error('create_neuron_data:regionEmpty', ...
        'Region %d has no allowed pixel left after intersecting with the mask.', ...
        find(areas_allowed == 0, 1));
end

% --- neurons per region ----------------------------------------------------
if isfield(opt,'region_n') && ~isempty(opt.region_n)
    n_per_region = expand_region_param(opt.region_n, R, 'region_n');
    n_per_region = round(n_per_region);
else
    % Not given: split opt.Nneu in proportion to the area actually available.
    n_per_region = round(opt.Nneu * areas_allowed / sum(areas_allowed));
end
if any(n_per_region < 0)
    error('create_neuron_data:regionN', 'region_n must be non-negative.');
end

% --- minimum separation per region ----------------------------------------
if isfield(opt,'region_min_dist') && ~isempty(opt.region_min_dist)
    min_per_region = expand_region_param(opt.region_min_dist, R, 'region_min_dist');
else
    min_per_region = repmat(opt.min_dist, 1, R);
end
if any(min_per_region <= 0)
    error('create_neuron_data:regionMinDist', 'region_min_dist must be positive.');
end

% --- coherence per region --------------------------------------------------
if isfield(opt,'region_corr') && ~isempty(opt.region_corr)
    region_corr = expand_region_param(opt.region_corr, R, 'region_corr');
else
    region_corr = zeros(1, R);          % no coherence requested
end

[X, Y, region_id] = get_random_points_regions(region_masks, n_per_region, ...
    min_per_region, opt.plotme);

if numel(X) < sum(n_per_region)
    warning('create_neuron_data:underfilled', ...
        'Requested %d neurons, placed %d.', sum(n_per_region), numel(X));
end

place = struct();
place.X = X;                                  % column index, full frame (opt.d)
place.Y = Y;                                  % row index, full frame (opt.d)
place.X_saved = X - crop;                     % column index in the saved video
place.Y_saved = Y - crop;                     % row index in the saved video
place.region_id = region_id;                  % 1..R
place.corr_strength = region_corr(region_id); % per-neuron corr_strength
place.corr_length = opt.corr_length;          % in gSig units
place.region_corr = region_corr;              % per-region corr_strength
place.region_masks_saved = masks_saved;       % logical masks, saved-frame geometry
place.region_id_image_saved = region_id_image;
place.region_crop = crop;                     % saved row r = full-frame row r+crop
place.region_areas_requested = areas;
place.region_areas_saved = areas_saved;       % achieved by grow_random_regions
place.region_areas_allowed = areas_allowed;   % after intersecting with the mask
place.n_per_region = n_per_region;
place.min_dist_per_region = min_per_region;
% kept for compatibility with the dense_region placement struct
place.dense_region_full  = [];
place.dense_region_saved = [];
end


function v = expand_region_param(v, R, name)
% Accept a scalar or a length-R vector, always return a 1-by-R row.
v = double(v(:))';
if isscalar(v)
    v = repmat(v, 1, R);
elseif numel(v) ~= R
    error('create_neuron_data:regionParamSize', ...
        '%s must be a scalar or have one entry per region (%d), got %d.', ...
        name, R, numel(v));
end
end


function g = footprint_gsig(As)
% Effective Gaussian width (in pixels) of a stack of neuron footprints,
% measured as the square root of the mean isotropic second spatial moment.
% For the footprint libraries shipped with the simulator this comes out at
% about 2.6 px, which is the same scale as the gSig CaliAli is run with.
As = double(As);
[h, w, n] = size(As);
[cc, rr] = meshgrid(1:w, 1:h);
rr = rr(:); cc = cc(:);
v = zeros(1, n);
for i = 1:n
    a = As(:,:,i);
    a = a(:) - min(a(:));
    tot = sum(a);
    if tot <= 0
        v(i) = NaN;
        continue
    end
    a = a / tot;
    my = sum(a .* rr);
    mx = sum(a .* cc);
    v(i) = sum(a .* ((rr - my).^2 + (cc - mx).^2)) / 2;   % per-axis variance
end
g = sqrt(mean(v(~isnan(v))));
if ~isfinite(g) || g <= 0
    error('create_neuron_data:gSig', 'Could not measure the footprint radius.');
end
end


function S = force_active_rows(S)
% Give every silent neuron one spike, mirroring the force_active branch of
% create_random_spike. correlated_spikes assigns round(P*F) spikes per neuron,
% so a neuron whose rate is below half a frame would otherwise be silent for
% the whole session.
silent = find(sum(S,2) == 0);
for i = 1:numel(silent)
    S(silent(i), randi(size(S,2), 1)) = 1;
end
end


function [X,Y,place] = place_heterogeneous(BW,opt)
% Place opt.Nneu neurons with a different density inside one or more
% rectangles ("dense regions") than outside them.
%
% opt.dense_region is given in SAVED-VIDEO pixel coordinates by default:
% Add_NRmotion drops 20 px on every border, so saved row r is full-frame row
% r+20. The offset is applied here, once, in dense_region_to_full.
%
% Region 1..R are the rectangles, region R+1 is everything else inside the
% heart mask. Rectangles are filled first, so the dense clumps get the pixels
% they need before the background competes for them.

d1 = size(BW,1);
d2 = size(BW,2);

[rect_full, rect_saved] = dense_region_to_full(opt, d1, d2);
R = size(rect_full,1);

% --- per-rectangle number of neurons -------------------------------------
frac = opt.dense_frac(:)';
if isscalar(frac)
    frac = repmat(frac/R, 1, R);   % scalar = TOTAL dense fraction, split evenly
elseif numel(frac) ~= R
    error('create_neuron_data:denseFracSize', ...
        'dense_frac must be a scalar or have one entry per dense_region row (%d).', R);
end
if any(frac < 0) || sum(frac) > 1 + 1e-9
    error('create_neuron_data:denseFracRange', ...
        'dense_frac entries must be non-negative and sum to at most 1 (got %.4g).', sum(frac));
end

n_rect = round(frac * opt.Nneu);
n_rect = min(n_rect, opt.Nneu);
n_bg   = opt.Nneu - sum(n_rect);
if n_bg < 0
    n_rect(end) = n_rect(end) + n_bg;
    n_bg = 0;
end

% --- per-rectangle minimum separation ------------------------------------
if isfield(opt,'dense_min_dist') && ~isempty(opt.dense_min_dist)
    dmd = opt.dense_min_dist(:)';
    if isscalar(dmd)
        dmd = repmat(dmd, 1, R);
    elseif numel(dmd) ~= R
        error('create_neuron_data:denseMinDistSize', ...
            'dense_min_dist must be a scalar or have one entry per dense_region row (%d).', R);
    end
else
    dmd = repmat(opt.min_dist, 1, R);   % no contrast requested, same separation
end

% --- masks ----------------------------------------------------------------
% Rectangles are made mutually exclusive: a pixel claimed by an earlier
% rectangle is removed from the later ones. Everything is intersected with the
% heart mask so the mask keeps being respected.
region_masks = cell(1, R+1);
claimed = false(d1, d2);
for r = 1:R
    m = false(d1, d2);
    m(rect_full(r,1):rect_full(r,2), rect_full(r,3):rect_full(r,4)) = true;
    m = m & BW & ~claimed;
    claimed = claimed | m;
    region_masks{r} = m;
end
region_masks{R+1} = BW & ~claimed;

n_per_region   = [n_rect, n_bg];
min_per_region = [dmd, opt.min_dist];

[X, Y, region_id] = get_random_points_regions(region_masks, n_per_region, ...
    min_per_region, opt.plotme);

if numel(X) < opt.Nneu
    warning('create_neuron_data:underfilled', ...
        'Requested %d neurons, placed %d.', opt.Nneu, numel(X));
end

place = struct();
place.X = X;                       % column index, full frame (opt.d)
place.Y = Y;                       % row index, full frame (opt.d)
place.X_saved = X - 20;            % column index in the saved video
place.Y_saved = Y - 20;            % row index in the saved video
place.region_id = region_id;       % 1..R dense rectangles, R+1 background
place.dense_region_full  = rect_full;
place.dense_region_saved = rect_saved;
place.n_per_region = n_per_region;
place.min_dist_per_region = min_per_region;
end


function [rect_full, rect_saved] = dense_region_to_full(opt, d1, d2)
% Convert opt.dense_region rows [r1 r2 c1 c2] to full-frame pixel indices.
rect = opt.dense_region;
if size(rect,2) ~= 4
    error('create_neuron_data:denseRegionShape', ...
        'dense_region must be a M x 4 matrix of [r1 r2 c1 c2] rows.');
end

frame = 'saved';
if isfield(opt,'dense_region_frame') && ~isempty(opt.dense_region_frame)
    frame = lower(char(opt.dense_region_frame));
end
switch frame
    case 'saved'
        off = 20;   % Add_NRmotion crops 20 px per border
    case 'full'
        off = 0;
    otherwise
        error('create_neuron_data:denseRegionFrame', ...
            'dense_region_frame must be ''saved'' or ''full''.');
end

rect_full  = round(rect) + off;
rect_saved = round(rect) + (off - 20);

% Clip to the frame and check the rectangle is not degenerate.
clipped = false;
for r = 1:size(rect_full,1)
    a = rect_full(r,:);
    a(1) = max(1, a(1));  a(2) = min(d1, a(2));
    a(3) = max(1, a(3));  a(4) = min(d2, a(4));
    if any(a ~= rect_full(r,:)); clipped = true; end
    if a(2) < a(1) || a(4) < a(3)
        error('create_neuron_data:denseRegionEmpty', ...
            'dense_region row %d is empty after clipping to the frame.', r);
    end
    rect_full(r,:) = a;
end
if clipped
    warning('create_neuron_data:denseRegionClipped', ...
        'dense_region was clipped to the simulated frame.');
end
end


%% Convolve with time constant
function out=conv_spikes(S,invtRise,invtDecay,F)
t = 0:F-1; %
for i=1:size(S,1)
    rise=1/lognrnd(invtRise(1),invtRise(2));
    decay=1/lognrnd(invtDecay(1),invtDecay(2));
    g = exp(-t/decay)-exp(-t/rise);
    temp=S(i,:);
    c = conv(temp, g);
    out(i,:)=c(1,1:F);
end
end

function S=fix_inactive(S)
d=size(S,2);
in=find(sum(S,2)==0);
for i=1:size(in,1)
    S(in(i),randi(d,1))=1;
end
end





