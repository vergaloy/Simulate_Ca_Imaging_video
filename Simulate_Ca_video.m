function [pth_out,outpath,Mot,file_name, A, GT_motion_xy, GT_nonrigid] = Simulate_Ca_video(varargin)
% Simulate_Ca_video generates synthetic calcium imaging videos with optional across-session misalignment, within-session motion, and remapping.
% It simulates neural activity, misalignment artifacts, baseline variability, and saves session-wise videos and metadata.
%
% Two separate things move the image, and every option name says which one:
%   across_session_*  one fixed warp per session, what ALIGNMENT has to undo
%   within_session_*  a different shift on every frame, what MOTION CORRECTION
%                     has to undo
% They are independent, so a session can be misaligned without shaking, shake
% without being misaligned, or both.

% Example usages:
% Simulate_Ca_video('across_session_nonrigid',0,'Nneu',100,'ses',2,'CA1_A',true,'PNR',1);
% Simulate_Ca_video('Nneu',300,'ses',20,'F',1500,...);
% V = Simulate_Ca_video('save_videos',false,raw_data{1,:});

% Parse inputs and initialize options
opt = int_var(varargin);
outpath = get_out_path(opt);  % choose output folder if saving enabled

%% Generate synthetic data
bl = create_baseline(opt);  % create session-specific background baselines
[A_GT, bA_GT, C, S, LFP, placement] = create_neuron_data(opt);  % simulate neurons and temporal traces
opt.Nneu2 = size(A_GT{1,1}, 3);  % store number of neurons

C = add_remapping_drifting(C, opt);  % apply remapping or activity drift if needed

% Measure the coherence that was actually achieved on the calcium traces. The
% requested region_corr is a LATENT correlation of the spike generator; binary
% thresholding lowers it and the calcium kernel raises it again, so the trace
% correlation has to be measured and stored, never assumed.
region_info = collect_region_info(opt, placement, C);

% Apply synthetic non-rigid misalignment to neuron components for each session
[BL(:,:,1), A{1,1}, bA{1,1}, Mot{1}] = Add_NRmotion(bl(:,:,1), A_GT{1}, bA_GT{1}, ...
    0, opt.plotme, opt.across_session_translation, opt.across_session_nonrigid_scale);
for i = 2:opt.ses
    [BL(:,:,i), A{1,i}, bA{1,i}, Mot{i}] = Add_NRmotion(bl(:,:,i), A_GT{i}, bA_GT{i}, ...
        opt.across_session_nonrigid, opt.plotme, opt.across_session_translation, opt.across_session_nonrigid_scale);
end
BL = v2uint8(BL);  % convert baseline to uint8

[d1, d2] = size(BL(:,:,1));  % get spatial dimensions

% Crop the ground-truth footprints to the field of view that is actually
% written to disk. Add_NRmotion discards 20 px on every border of the warped
% data, so an uncropped A_GT/bA_GT would not share a coordinate frame with the
% session videos, the GT video, or the misaligned footprints in A/bA.
for i = 1:opt.ses
    A_GT{i}  = A_GT{i}(21:end-20, 21:end-20, :);
    bA_GT{i} = bA_GT{i}(21:end-20, 21:end-20, :);
end

GT_motion_xy = cell(1, opt.ses);  % store within-session motion ground truth
% Within-session non-rigid ground truth: the control points and their
% trajectories, which is everything needed to rebuild the field that was applied.
GT_nonrigid  = cell(1, opt.ses);

%% Simulate each session one-by-one (memory-efficient)
file_name = datestr(now, 'yymmdd_HHMMSS');  % timestamp-based base filename
ix1 = 1; F = opt.F; ix2 = F;
PNR = opt.PNR; mult = 0.8;

saveVideos = logical(opt.save_videos);
saveMat = logical(opt.save_mat);
saveGT = logical(opt.save_GT);

if saveGT
    % Prepare memory-mapped file for writing GT frames incrementally
    matGT = matfile([file_name '_GT.mat'], 'Writable', true);
end

pth_out = cell(1, opt.ses);
batch_sz = opt.batch_sz;

% Within-session motion amplitude, per session. A scalar applies to all of them,
% which is what this used to be; a vector gives each session its own. Sessions
% recorded on different days do not shake by the same amount, and a pipeline that
% corrects each session on its own then crops each to its own valid region, so
% equal amplitudes produce sessions that differ in size by a pixel or two and
% never exercise the case where they differ properly.
motion_std = opt.within_session_translation_std(:)';
if isscalar(motion_std)
    motion_std = repmat(motion_std, 1, opt.ses);
elseif numel(motion_std) ~= opt.ses
    error('Simulate_Ca_video:motionStd', ...
        ['within_session_translation_std must be a scalar or one value per session. ' ...
         'Got %d values for %d sessions.'], numel(motion_std), opt.ses);
end

% Within-session NON-RIGID amplitude, same rules. Translation moves the whole
% frame; this bends it, so a piecewise-rigid corrector has something real to
% find. Zero, the default, leaves the motion purely rigid as before.
nr_std = opt.within_session_nonrigid_std(:)';
if isscalar(nr_std)
    nr_std = repmat(nr_std, 1, opt.ses);
elseif numel(nr_std) ~= opt.ses
    error('Simulate_Ca_video:nonrigidStd', ...
        ['within_session_nonrigid_std must be a scalar or one value per session. ' ...
         'Got %d values for %d sessions.'], numel(nr_std), opt.ses);
end

for i = progress(1:opt.ses)
    std_i = motion_std(i);
    % Precompute within-session motion, if any
    if std_i > 0
        prev_rng = rng;
        if isempty(opt.within_session_motion_seed)
            seed_i = opt.seed + i;
        else
            seed_i = opt.within_session_motion_seed + i - 1;
        end
        [dx, dy] = make_xy_motion_ar(F, opt.within_session_motion_smoothness, std_i, ...
            opt.within_session_motion_max, seed_i);
        rng(prev_rng);
        GT_motion_xy{i} = {dx, dy};
    else
        dx = [];
        dy = [];
        GT_motion_xy{i} = [];
    end

    % Precompute within-session non-rigid motion, if any. The spline basis
    % depends only on where the control points sit, so it is built once per
    % session and reused for every frame.
    nr_i = nr_std(i);
    if nr_i > 0
        prev_rng = rng;
        if isempty(opt.within_session_motion_seed)
            nr_seed = opt.seed + 1000 + i;
        else
            nr_seed = opt.within_session_motion_seed + 1000 + i - 1;
        end
        [ctrl_xy, cdx, cdy] = make_nonrigid_motion_ar([d1, d2], F, ...
            opt.within_session_nonrigid_grid, opt.within_session_motion_smoothness, nr_i, opt.within_session_motion_max, nr_seed);
        nr_basis = tps_dense_basis(ctrl_xy, [d1, d2]);
        rng(prev_rng);
        GT_nonrigid{i} = {ctrl_xy, cdx, cdy};
    else
        cdx = []; cdy = []; nr_basis = [];
        GT_nonrigid{i} = [];
    end

    % Session output setup
    if std_i==0 && nr_i==0
        mc_tag='_mc';
    else
        mc_tag='';
    end
    if saveVideos
        if opt.save_avi
            pth_out{i}=[file_name '_ses' sprintf('%02d', i-1) mc_tag '.avi'];
            v = VideoWriter(pth_out{i}, 'Uncompressed AVI');
            v.FrameRate = 25;
            open(v);
        else
            pth_out{i}=[file_name '_ses' sprintf('%02d', i-1) mc_tag];
        end
    else
        pth_out{i} = [];
    end

    % Already cropped to the saved field of view, once for every session,
    % where A_GT is built. Cropping here as well would do it F/batch_sz times.
    tA = A_GT{i};
    tbA = bA_GT{i};
    session_frames = ix1:ix2;

    if saveGT && i == 1
        matGT.GT = zeros(d1, d2, F * opt.ses, 'uint8');  % preallocate on disk
    end

    for start_idx = progress(1:batch_sz:F)
        batch_frames = start_idx:min(start_idx + batch_sz - 1, F);
        global_idx = session_frames(batch_frames);

        % --- Generate simulated session data in batches ---
        V_batch = v2uint8(reshape(reshape(A{i}, d1*d2, []) * C(:, global_idx), d1, d2, []) + ...
            reshape(reshape(bA{i}, d1*d2, []) * C(:, global_idx), d1, d2, []));

        % Add smoothed noise
        N_batch = rand(d1, d2, numel(batch_frames), 'single');
        N_batch = imgaussfilt(N_batch);
        N_batch = N_batch - mean(N_batch, 3);
        N_batch = v2uint8(N_batch);

        % Combine neural signal, noise, and baseline
        FV_batch = mult * BL(:, :, i) + ...
            (1 - mult) * V_batch * (1 / (1 / PNR + 1)) + ...
            (1 - mult) * N_batch * (1 - 1 / (1 / PNR + 1));
        FV_batch = v2uint8(FV_batch);

        % Apply within-session translational motion if requested
        if std_i > 0
            FV_batch = apply_xy_translation(single(FV_batch), dx(batch_frames), dy(batch_frames));
            FV_batch = v2uint8(FV_batch);
        end

        % Then bend it. Translation first, deformation second, which is the order
        % a correction pipeline undoes them in: a whole-frame shift, then what is
        % left over.
        if nr_i > 0
            FV_batch = apply_nonrigid_motion(single(FV_batch), nr_basis, ...
                cdx(:, batch_frames), cdy(:, batch_frames));
            FV_batch = v2uint8(FV_batch);
        end

        % --- Save session output video ---
        if saveVideos
            if opt.save_avi
                for k = 1:size(FV_batch, 3)
                    writeVideo(v, FV_batch(:, :, k) + 1);
                end
            else
                saveash5(FV_batch + 1, pth_out{i});
            end
        end

        % --- Ground truth generation ---
        if saveGT
            V_GT_batch = v2uint8(reshape(reshape(tA, d1*d2, []) * C(:, global_idx), d1, d2, []) + ...
                reshape(reshape(tbA, d1*d2, []) * C(:, global_idx), d1, d2, []));
            matGT.GT(:, :, global_idx) = V_GT_batch;  % memory-safe write
        end

        % Clean batch variables from memory
        clear V_batch N_batch FV_batch V_GT_batch
    end

    if saveVideos && opt.save_avi
        close(v);
        pth_out{i}=which([file_name '_ses' sprintf('%02d', i-1) mc_tag '.avi']);
    elseif saveVideos && ~opt.save_avi
        pth_out{i}=which([file_name '_ses' sprintf('%02d', i-1) mc_tag]);
    end

    % Update frame index
    ix1 = ix1 + F;
    ix2 = ix2 + F;

    % Clean session variables from memory
    clear tA tbA dx dy
end

%% Final saving
if saveGT
    % Save full GT video
    if opt.save_avi
        save_as_avi(matGT.GT + 1, [file_name '_GT.avi']);
    else
        saveash5(matGT.GT + 1, [file_name '_GT']);
    end
end

if saveMat
    % Save simulation metadata (everything else)
    save([file_name '_meta.mat'], 'A_GT', 'A', 'C', 'S', 'BL', 'opt', 'Mot', 'LFP', 'GT_motion_xy', 'GT_nonrigid', 'placement', 'region_info', '-v7.3');
end

if ~(saveVideos || saveMat || saveGT)
    file_name = [];
end

end
function region_info = collect_region_info(opt, placement, C)
% Everything the coherence analysis needs, gathered in one place: which region
% each neuron belongs to, the region masks, the coherence that was requested,
% and the coherence that was actually achieved on the calcium traces.
%
% Returns [] when the region layout was not used, so that a run without
% region_areas is unaffected.

region_info = [];
if ~isfield(opt,'region_areas') || isempty(opt.region_areas)
    return
end

region_info = struct();
region_info.region_id           = placement.region_id(:)';   % 1..R per neuron
region_info.region_masks_saved  = placement.region_masks_saved;
region_info.region_id_image_saved = placement.region_id_image_saved;
region_info.region_crop         = placement.region_crop;     % saved row r = full row r+crop
region_info.corr_strength       = placement.corr_strength(:)'; % per neuron
region_info.region_corr         = placement.region_corr;     % per region
region_info.corr_length         = placement.corr_length;     % in gSig units
region_info.gSig_fullres        = placement.gSig_fullres;    % pixels
region_info.corr_length_px      = placement.corr_length * placement.gSig_fullres;
region_info.region_areas_requested = placement.region_areas_requested;
region_info.region_areas_saved     = placement.region_areas_saved;
region_info.region_areas_allowed   = placement.region_areas_allowed;
region_info.n_per_region        = placement.n_per_region;
region_info.min_dist_per_region = placement.min_dist_per_region;

% Neuron centres in the saved frame, and the pairwise distances the law is
% fitted against.
X = placement.X_saved(:);
Y = placement.Y_saved(:);
region_info.X_saved = X';
region_info.Y_saved = Y';
region_info.pair_distance = hypot(X - X.', Y - Y.');

% ACHIEVED pairwise calcium trace correlation. This is what the coherence law
% is fitted on. It is NOT the spike correlation and it is NOT region_corr.
region_info.trace_corr = pair_corr(C);

% Per session as well, since remapping can silence a neuron in some sessions.
F = opt.F;
region_info.trace_corr_per_session = cell(1, opt.ses);
for i = 1:opt.ses
    region_info.trace_corr_per_session{i} = pair_corr(C(:, (i-1)*F+1:i*F));
end
end

function Rc = pair_corr(C)
% Pearson correlation between every pair of rows, with constant rows (a neuron
% silenced by remapping) reported as NaN rather than 0.
C = double(C);
Cc = C - mean(C, 2);
sd = sqrt(sum(Cc.^2, 2));
bad = sd <= 0;
sd(bad) = 1;
Cn = Cc ./ sd;
Rc = Cn * Cn.';
Rc(bad, :) = NaN;
Rc(:, bad) = NaN;
Rc(1:size(Rc,1)+1:end) = 1;
Rc(bad, bad) = NaN;
end

function opt=int_var(varargin)
%% INTIALIZE VARIABLES
inp = inputParser;
valid_v = @(x) isnumeric(x);
valid_bool = @(x) islogical(x) || (isnumeric(x) && isscalar(x) && ismember(x,[0,1]));
valid_batch = @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x > 0);
addParameter(inp, 'outpath', [])               % Output path where simulation results will be saved
addParameter(inp, 'min_dist', 8, valid_v)      % Minimum distance between neurons in the simulated data
% --- spatially heterogeneous neuron density (optional, off by default) ---
% dense_region: M x 4 matrix, one [r1 r2 c1 c2] rectangle per row, inclusive
%   pixel indices. By default the rectangle is read in SAVED-VIDEO coordinates,
%   i.e. the frame actually written to disk, which is 40 px smaller than 'd' on
%   each axis because Add_NRmotion crops 20 px per border. Leave [] (default)
%   for the original homogeneous placement.
% dense_frac: fraction of Nneu placed inside the rectangles. A scalar is the
%   TOTAL dense fraction and is split evenly between the rectangles; a vector
%   gives the fraction of each rectangle. The remainder goes to the background.
% dense_min_dist: minimum neuron separation inside the rectangles (scalar or one
%   per rectangle). [] means "same as min_dist", i.e. no density contrast.
%   Outside the rectangles min_dist is used, as before.
% dense_region_frame: 'saved' (default) or 'full', the coordinate frame the
%   rectangles are expressed in.
addParameter(inp, 'dense_region', [], @(x) isempty(x) || (isnumeric(x) && size(x,2)==4))
addParameter(inp, 'dense_frac', 0.5, valid_v)
addParameter(inp, 'dense_min_dist', [], @(x) isempty(x) || isnumeric(x))
addParameter(inp, 'dense_region_frame', 'saved', @(x) ischar(x) || isstring(x))
% placement_margin: forbid neuron centres within this many pixels of the
%   FULL-frame ('d') border. [] (default) keeps the original behaviour. Only
%   useful with create_mask=false, where neurons would otherwise be placed in
%   the 20 px strip that Add_NRmotion crops away, leaving empty ground-truth
%   footprints. 45 px (20 crop + half of the 50 px neuron kernel) is a safe value.
addParameter(inp, 'placement_margin', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x)))
% --- random continuous regions with per-region density and activity
%     coherence (optional, off by default) --------------------------------
% region_areas: 1xR target region areas, in SAVED-FRAME pixels, i.e. on the
%   frame actually written to disk, which is 'd' minus 40 on each axis because
%   Add_NRmotion crops 20 px per border. Same coordinate convention as the
%   'saved' setting of dense_region_frame. Because the regions must jointly
%   cover the frame, only the RATIOS between the entries can be honoured and
%   grow_random_regions rescales them to sum to the saved-frame area. Leave []
%   (default) for the original homogeneous placement and independent spikes.
% region_n: 1xR number of neurons per region. [] splits Nneu in proportion to
%   the area available in each region.
% region_min_dist: 1xR minimum neuron separation per region, in pixels.
%   [] uses min_dist everywhere.
% region_corr: 1xR zero-distance activity correlation of each region, in
%   [0, 1). Every neuron inherits the value of the region it was placed in and
%   the spike trains come from correlated_spikes instead of the independent
%   draw in create_random_spike. [] means zero, i.e. no coherence.
% corr_length: correlation decay length, in gSig units. The pairwise latent
%   correlation is region_corr * exp(-distance / (corr_length * gSig)), with
%   gSig measured from the neuron footprints actually drawn (about 2.6 px at
%   scale 1).
% region_seed: seed for the region layout only. [] falls back to 'seed'. The
%   layout draw restores the global random state, so it never changes the rest
%   of the simulation.
addParameter(inp, 'region_areas', [], @(x) isempty(x) || (isnumeric(x) && isvector(x)))
addParameter(inp, 'region_n', [], @(x) isempty(x) || isnumeric(x))
addParameter(inp, 'region_min_dist', [], @(x) isempty(x) || isnumeric(x))
addParameter(inp, 'region_corr', [], @(x) isempty(x) || isnumeric(x))
addParameter(inp, 'corr_length', 3, @(x) isnumeric(x) && isscalar(x) && x > 0)
% burst_tau: width in FRAMES of a Gaussian kernel applied along time to the
%   latent field before it is thresholded into spikes. 0 (default) reproduces
%   the previous behaviour exactly. Larger values make spikes CLUSTER into
%   bursts, which is what lets the achieved CALCIUM TRACE correlation reach
%   useful values: thresholding a Gaussian into isolated spikes destroys most
%   of the latent correlation and caps the trace correlation near 0.12 even at
%   corr_strength 0.8. The smoothing is identical for every neuron, so the
%   spatial correlation structure is preserved exactly.
addParameter(inp, 'burst_tau', 0, @(x) isnumeric(x) && isscalar(x) && x >= 0)
% region_layout: 'grown' (default, irregular regions grown to region_areas) or
%   'quadrants' (a centred square window split 2x2, equal areas). Use
%   'quadrants' when density and correlation must be CROSSED: the grown layout
%   varies area and shape alongside them, which confounds the two factors.
% window_side: optional side of that square in pixels; defaults to the largest
%   square fitting inside placement_margin.
addParameter(inp, 'region_layout', 'grown', @(x) ischar(x) || isstring(x))
addParameter(inp, 'window_side', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x)))
addParameter(inp, 'region_seed', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x)))
addParameter(inp, 'Nneu', 50, valid_v)         % Number of neurons (or neural units) to be simulated
addParameter(inp, 'PNR', 2, valid_v)          % Peak-to-Noise Ratio (signal-to-noise ratio) for the simulated data
addParameter(inp, 'd', [220, 300], valid_v)    % Dimensions of the simulated video frames (width and height) in pixels
addParameter(inp, 'F', 1000, valid_v)          % Frame rate (frames per second) for the simulated calcium video data in each session
addParameter(inp, 'batch_sz', [], valid_batch) % Frames per batch (defaults to F)
addParameter(inp, 'overlap', 1, valid_v)       % Proportion of neurons remapping across multiple sessions
addParameter(inp, 'overlapMulti', 0, valid_v)  % Remapping strenght. 0 means 100% remapping (should be the opposite, sorry!)
% --- ACROSS-SESSION misalignment ------------------------------------------
% One fixed warp per session, applied to the whole recording by Add_NRmotion.
% This is what the inter-session ALIGNMENT stage has to undo. It does not vary
% from frame to frame. Session 1 is the reference and gets no non-rigid warp.
% across_session_nonrigid: amplitude of the smooth warp, in pixels. 0 = none.
% across_session_nonrigid_scale: how broad that warp is, as the Gaussian sigma
%   in pixels used to smooth the random field. Large values bend the frame in
%   one slow sweep, small values ripple it.
% across_session_translation: length in pixels of the rigid shift between
%   sessions. The direction is random, the length is this. 0 = no shift.
addParameter(inp, 'across_session_nonrigid', 0, valid_v)
addParameter(inp, 'across_session_nonrigid_scale', 60, valid_v)
addParameter(inp, 'across_session_translation', 1, valid_v)
% --- WITHIN-SESSION motion -------------------------------------------------
% Moves EVERY FRAME of a session on its own. This is what MOTION CORRECTION
% has to undo, and it is independent of the across-session misalignment above.
% The trajectory is an AR(1) walk, so consecutive frames move together.
% within_session_translation_std: std in pixels of the whole-frame shake.
%   Scalar for every session, or one value per session so they differ.
% within_session_nonrigid_std: std in pixels of the BEND applied on top of that
%   shake, which is what a piecewise-rigid corrector has to find. 0 = the
%   motion stays purely rigid.
% within_session_nonrigid_grid: control points per axis of the bend. 3 = 3x3.
% within_session_motion_smoothness: AR(1) coefficient, shared by the shake and
%   the bend. Near 1 gives slow drift, near 0 gives frame-to-frame jitter.
% within_session_motion_max: optional cap in pixels on peak displacement.
% within_session_motion_seed: seed offset, for a reproducible trajectory.
addParameter(inp, 'within_session_translation_std', 3, valid_v)
addParameter(inp, 'within_session_nonrigid_std', 0, valid_v)
addParameter(inp, 'within_session_nonrigid_grid', 3, valid_v)
addParameter(inp, 'within_session_motion_smoothness', 0.99, valid_v)
addParameter(inp, 'within_session_motion_max', [], @(x) isnumeric(x) || isempty(x))
addParameter(inp, 'within_session_motion_seed', [], valid_v)
addParameter(inp, 'ses', 2, valid_v)           % Number of sessions to be generated
addParameter(inp, 'seed', 'shuffle')           % Random number generator seed ('shuffle' for a new seed each run)
addParameter(inp, 'B', '1')                    % Baseline id (1-8). Different valeus used different baseline images
addParameter(inp, 'spike_prob', [-4.9, 2.25])  % Probability distribution parameters for spike events (log scale mean and std)
addParameter(inp, 'save_videos', true, valid_bool)          % Flag to save session videos (true/false)
addParameter(inp, 'create_mask', true)         % Flag to create a mask during the simulation (true/false)
addParameter(inp, 'comb_fact', 0, valid_v)     % Combination factor used in the simulation process
addParameter(inp, 'drift', 0, valid_v)         % Presence of drifting activities in the simulated data (0 or 1)
addParameter(inp, 'LFP', 0, valid_v)           % Flag to simulate Local Field Potentials (LFP) data (0 or 1)
addParameter(inp, 'sf', 10, valid_v)           % Scaling factor used in the simulation process
addParameter(inp, 'plotme', 0, valid_v)        % Flag to generate plots during simulation (0 or 1)
addParameter(inp, 'invtRise', [2.08, 0.29], valid_v)   % Parameters related to the rising phase of calcium signals (log scale)
addParameter(inp, 'invtDecay', [0.55, 0.44], valid_v) % Parameters related to the decaying phase of calcium signals (log scale)
addParameter(inp, 'disappearBV', 0, valid_v)   % Erode BV
addParameter(inp, 'CA1_A', false)              % Change neurons shapes across sessions. Utilize CA1 data
addParameter(inp, 'A2', false)                 % Use CA1 neurons shapes instead of DG.
addParameter(inp, 'force_active', 1)           % All nuerons should have at least one Calcium transient per session (unless remapping)
addParameter(inp, 'save_avi', 1)               % Save as avi. otherwise save as .h5
addParameter(inp, 'scale', 1)                  % Resize the dimensions of A
addParameter(inp, 'save_mat', 1, valid_bool)   % Whether to save metadata (.mat) files
addParameter(inp, 'save_GT', 1, valid_bool)    % Whether to save GT video
varargin=varargin{1, 1};

reject_renamed_options(varargin);

inp.KeepUnmatched = true;
parse(inp,varargin{:});
opt=inp.Results;
opt.B=str2num(string(opt.B));

if isfield(inp.Unmatched,'save_files') && any(strcmpi(inp.UsingDefaults,'save_videos'))
    opt.save_videos = logical(inp.Unmatched.save_files);
end

opt.save_videos = logical(opt.save_videos);
opt.save_mat = logical(opt.save_mat);
opt.save_GT = logical(opt.save_GT);
if isempty(opt.batch_sz)
    opt.batch_sz = opt.F;
end
opt.batch_sz = max(1, min(opt.F, round(opt.batch_sz)));


s=rng(opt.seed,'twister');
s=rng(opt.seed,'twister');
opt.seed=s.Seed;
while 1
    s=rng(opt.seed,'twister');
    if s.Seed==opt.seed
        break
    end
end
s.Seed

end

function  outpath=get_out_path(opt)
if opt.save_videos || opt.save_mat || opt.save_GT
    if isempty(opt.outpath)
        outpath = uigetdir(pwd,'select output folder');
    else
        outpath=opt.outpath;
    end
    cd(outpath);
else
    outpath=[];
end
end

function bl=create_baseline(opt)
comb_fact=1-opt.comb_fact;
ses=opt.ses;
load('BL.mat','Mr');

if length(opt.B)==1
    Mr=Mr{opt.B,1};
    bl=zeros(opt.d(1),opt.d(2),2);
    for i=1:size(Mr,3)
        bl(:,:,i)=imresize(Mr(:,:,i),opt.d);
    end
    if ses>2
        p=rand(1,ses);
        temp=bl;
        bl=zeros(opt.d(1),opt.d(2),ses);
        for i=1:ses
            bl(:,:,i)=temp(:,:,1)*p(i)+temp(:,:,2)*(1-p(i))+randn/10;
        end
        bl=mat2gray(bl);
    end
else
    temp=cat(3,imresize(Mr{opt.B(1),1}(:,:,1),opt.d),imresize(Mr{opt.B(2),1}(:,:,1),opt.d));
    p=linspace(1,comb_fact,ses);

    for i=1:ses
        bl(:,:,i)=temp(:,:,1)*p(i)+temp(:,:,2)*(1-p(i));
    end

end

if opt.disappearBV>0
    for i=1:size(bl,3)
        im=bl(:,:,i);
        gs=medfilt2(mean(im,2)*mean(im,1),[25 25])+rescale(randn(size(im)),min(im,[],'all'),max(im,[],'all'))./10;
        im=rescale(im*(1-opt.disappearBV)+gs*opt.disappearBV, ...
            min(bl(:,:,i),[],'all'),max(bl(:,:,i),[],'all'));
        bl(:,:,i)=im;
    end
else
    bl=mat2gray(bl);
end


end

function C=add_remapping_drifting(C,opt)
Nneu2=opt.Nneu2;
ses=opt.ses;
F=opt.F;
if ses>1
    temp=double(rand(Nneu2,ses)<=opt.overlap);
    t2=repelem(eye(ses),ceil(Nneu2/ses),1);
    t2(randperm(size(t2,1),size(t2,1)-size(temp,1)),:)=[];
    temp=(temp+t2)>0;
    M=double(repelem(temp,1,F));
    M(M==0)=opt.overlapMulti;
else
    M=ones(Nneu2,F);
end
C=C.*M;

if opt.drift==1
    M=sim_drifting_activities(Nneu2,ses);
    M=double(repelem(M,1,F));
    C=C.*M;
end
end

function [dx,dy] = make_xy_motion_ar(T, phi, std_px, max_px, seed)
% AR(1) zero-mean x/y displacement (pixels)
if nargin<2, phi=0.95; end                     % temporal smoothness (0..1)
if nargin<3, std_px=2.0; end                   % target stationary std (px)
if nargin<4, max_px=[]; end                    % optional peak cap
if nargin<5, rng('default'); else, rng(seed); end

sigma = std_px*sqrt(1-phi^2);                  % innovation std for AR(1)
e1 = sigma*randn(1,T); e2 = sigma*randn(1,T);
dx = filter(1,[1 -phi], e1);                   % x_t = phi*x_{t-1} + e_t
dy = filter(1,[1 -phi], e2);
dx = dx - mean(dx); dy = dy - mean(dy);        % center at 0

if ~isempty(max_px)
    s = max(1, max([abs(dx) abs(dy)],[],'all')/max_px);
    dx = dx./s; dy = dy./s;
end
end

function Yout = apply_xy_translation(Y, dx, dy)
% Y: [Ly x Lx x T], dx/dy: pixels
[~,~,T] = size(Y);
Yout = zeros(size(Y),'like',Y);
for k=1:T
    Yout(:,:,k) = imtranslate(Y(:,:,k), [dx(k) dy(k)], 'linear', ...
        'OutputView','same','FillValues',0);
end
end
function reject_renamed_options(args)
%% Stop a script that still uses the old motion and misalignment names.
%
% The old names did not say WHICH of the two effects they controlled.
% NR_misalignment and session_nonrigid_std were both non-rigid,
% translation_misalignment and session_motion_std were both translations, and
% motion_sz said "motion" while it sized an across-session warp. Every name now
% starts with the timescale it acts on: across_session_ or within_session_.
%
% This has to be checked by hand because KeepUnmatched is on, so an old name
% would otherwise be accepted, silently ignored, and the simulation would run
% with the default instead of the value that was asked for.

renamed = {
    'NR_misalignment',          'across_session_nonrigid'
    'motion_sz',                'across_session_nonrigid_scale'
    'translation_misalignment', 'across_session_translation'
    'session_motion_std',       'within_session_translation_std'
    'session_nonrigid_std',     'within_session_nonrigid_std'
    'nonrigid_grid',            'within_session_nonrigid_grid'
    'session_motion_phi',       'within_session_motion_smoothness'
    'session_motion_max',       'within_session_motion_max'
    'session_motion_seed',      'within_session_motion_seed'
    };

for k = 1:2:numel(args)
    if ~(ischar(args{k}) || isstring(args{k})), continue; end
    hit = strcmpi(char(args{k}), renamed(:,1));
    if any(hit)
        error('Simulate_Ca_video:renamedOption', ...
            ['"%s" was renamed to "%s". Every motion option now starts with ' ...
             'the timescale it acts on: across_session_ for the fixed warp ' ...
             'between sessions, within_session_ for the frame-to-frame ' ...
             'motion inside one session.'], ...
            renamed{hit,1}, renamed{hit,2});
    end
end
end
