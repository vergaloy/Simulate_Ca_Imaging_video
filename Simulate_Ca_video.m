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
%
% Optional, all off by default: motion whose character changes over time
% (rigid-only and bend stretches, local bends, sudden jumps; make_motion_regimes)
% and a camera model that integrates each pixel over a band-limited scene
% (camera_supersample; camera_setup). With any of them on, the within-session
% motion is composed into one field per frame and applied once
% (apply_camera_motion); with all of them off nothing changes, bit for bit.
%
% Also optional, all off by default: what the camera adds that does not move
% with the tissue (make_camera_optics; options in the "Camera optics and
% sensor" block): hot pixels, the sensor's fixed pattern and dust (path M3),
% vignetting from a real recording and an out-of-focus layer with a motion of
% its own (path M4), and brightness flicker and slow trends (path M6). Any of
% them takes the camera path; their ground truth is in GT_optics.
%
% Also optional: sensor noise (sensor_noise), white noise added to each frame
% after the motion, before the camera rounds it, so it does not move with the
% tissue; and
% scene_noise = false, which leaves the per-frame noise out of the scene. Both
% together move the noise from the scene to the sensor.
%
% Also optional: impacts (impacts, impact_*; make_impacts), where the animal
% knocks the microscope and the view is lost for about a second: the tissue's
% structure slides away with motion blur and blends into the structure of a
% different field of view (another baseline image) over the baseline's fixed
% low-pass part, then comes back. They take the camera path.
%
% Also optional: slow drift (slow_drift*; make_slow_drift), a session-long
% deformation set as a rate and composed with the rest of the motion; jolts
% (jolt*; make_jolts), a fast displacement of the same scene with motion blur,
% defocus and a brightness change; and acquisition glitches (glitch*;
% make_glitches), dropped, white, black and partial frames. Drift and jolts
% take the camera path.
%
% Also optional: a smooth background (smooth_background), where the scene (the
% baseline, the neurons, the background sources) stays in floating point all
% the way to the camera instead of being rounded to 8 bits first, which gives
% every smooth contour of the scene flat terraces; and activity bursts
% (bursts, burst_*; make_bursts), periods of strong synchronised activity
% (epileptic-like) where many neurons fire together for seconds and the
% background rises with them, with no motion.

% Example usages:
% Simulate_Ca_video('across_session_nonrigid',0,'Nneu',100,'ses',2,'CA1_A',true,'PNR',1);
% Simulate_Ca_video('Nneu',300,'ses',20,'F',1500,...);
% V = Simulate_Ca_video('save_videos',false,raw_data{1,:});

% Parse inputs and initialize options
opt = int_var(varargin);
outpath = get_out_path(opt);  % choose output folder if saving enabled

%% Generate synthetic data
bl = create_baseline(opt);  % create session-specific background baselines
[A_GT, bA_GT, C, S, LFP, placement, Cb] = create_neuron_data(opt);  % simulate neurons, background and traces
opt.Nneu2 = size(A_GT{1,1}, 3);  % store number of neurons

C = add_remapping_drifting(C, opt);  % apply remapping or activity drift if needed

% Measure the coherence that was actually achieved on the calcium traces. The
% requested region_corr is a LATENT correlation of the spike generator; binary
% thresholding lowers it and the calcium kernel raises it again, so the trace
% correlation has to be measured and stored, never assumed.
region_info = collect_region_info(opt, placement, C);

% Optional slow cyclic baseline, added after the coherence is measured so that
% region_info keeps describing the transients. Off unless slow_amp > 0.
[C, C_slow] = add_slow_baseline(C, opt, opt.slow_amp, 2000);

% Background traces. Under 'per_cell' each halo follows its own cell, as it
% always did; otherwise the background has traces of its own, and an optional
% slow cycle of its own (slow_bg_amp), independent of the neurons'.
if isempty(Cb)
    Cb = C;
    Cb_slow = C_slow;
else
    [Cb, Cb_slow] = add_slow_baseline(Cb, opt, opt.slow_bg_amp, 2500);
end

% Activity bursts (make_bursts), from a stream of their own per session
% (seed + 11000 + session). They are kept apart from C, which with Cb sets the
% movie's grey scale, and added to the image on that same scale, so the frames
% outside bursts are what they would be without them. The neurons' whole
% activity is C + C_burst.
[use_bursts, GT_bursts, C_burst] = session_bursts(C, placement, opt);

% Apply synthetic non-rigid misalignment to neuron components for each session
[BL(:,:,1), A{1,1}, bA{1,1}, Mot{1}] = Add_NRmotion(bl(:,:,1), A_GT{1}, bA_GT{1}, ...
    0, opt.plotme, opt.across_session_translation, opt.across_session_nonrigid_scale);
for i = 2:opt.ses
    [BL(:,:,i), A{1,i}, bA{1,i}, Mot{i}] = Add_NRmotion(bl(:,:,i), A_GT{i}, bA_GT{i}, ...
        opt.across_session_nonrigid, opt.plotme, opt.across_session_translation, opt.across_session_nonrigid_scale);
end
if opt.smooth_background
    BL = to_grey(BL);  % the same grey scale, not rounded
else
    BL = v2uint8(BL);  % convert baseline to uint8
end

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

% Regimes, local bends, jumps and the camera model (see the header). The path
% below is taken only when one of them is on.
rp = regime_options(opt);
ip = impact_options(opt);
use_impacts = ip.n > 0 || ~isempty(ip.onset);
use_camera = opt.camera_supersample > 1 || ~isempty(rp.rigid_frames) || ...
    rp.local_bends > 0 || rp.jumps > 0 || use_impacts;
GT_regimes = cell(1, opt.ses);
GT_impacts = cell(1, opt.ses);
if use_impacts
    % The foreign field of view: the high-pass part of another baseline image,
    % on the saved frame and on BL's grey scale. Loading it draws nothing.
    Hf0 = foreign_structure(opt, [d1, d2]);
end
% Slow drift, jolts and glitches (all optional, off; see the options).
use_drift = logical(opt.slow_drift);
use_jolts = opt.jolts > 0 || ~isempty(opt.jolt_onset);
use_glitches = opt.glitches > 0;
use_camera = use_camera || use_drift || use_jolts;
GT_drift = cell(1, opt.ses); GT_jolts = cell(1, opt.ses); GT_glitches = cell(1, opt.ses);
% The camera's optics and sensor (make_camera_optics), off by default.
op_p = optics_options(opt);
use_optics = op_p.hot_pixels || op_p.sensor_pattern || op_p.dust || ...
    op_p.vignetting || op_p.oof_layer || op_p.flicker || op_p.brightness_trend || ...
    any(op_p.black_level(:) > 0) || any(op_p.adc_gain(:) ~= 1);
use_camera = use_camera || use_optics;
GT_optics = cell(1, opt.ses);

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

    % Regimes, local bends and jumps, from a stream of their own, and the
    % camera that applies all the within-session motion in one go.
    cam_i = [];
    moves = std_i > 0 || nr_i > 0;
    if use_camera
        if isempty(opt.within_session_motion_seed)
            r_seed = opt.seed + 3000 + i;
        else
            r_seed = opt.within_session_motion_seed + 3000 + i - 1;
        end
        R_i = make_motion_regimes([d1, d2], F, rp, r_seed);
        R_i.camera_supersample = max(1, round(opt.camera_supersample));
        mo = struct('dx', zeros(1, F), 'dy', zeros(1, F), ...
            'ctrl_xy', [], 'cdx', [], 'cdy', [], 'R', R_i);
        if std_i > 0, mo.dx = dx; mo.dy = dy; end
        if nr_i > 0, mo.ctrl_xy = ctrl_xy; mo.cdx = cdx; mo.cdy = cdy; end
        cam_i = camera_setup([d1, d2], R_i.camera_supersample, mo);
        GT_regimes{i} = R_i;
        moves = moves || ~isempty(R_i.bumps) || ~isempty(R_i.jumps);
        if use_impacts
            % From a stream of their own (seed + 5000 + session).
            if isempty(opt.within_session_motion_seed)
                i_seed = opt.seed + 5000 + i;
            else
                i_seed = opt.within_session_motion_seed + 5000 + i - 1;
            end
            I_i = make_impacts(F, ip, i_seed);
            GT_impacts{i} = I_i;
            moves = moves || any(I_i.mask);
            % The baseline's low-pass part, fixed to the optics (BL's scale).
            LB0 = imgaussfilt(double(mult * BL(:, :, i)), opt.impact_cutoff);
        end
        if use_drift
            Dr_i = make_slow_drift([d1, d2], F, slow_drift_options(opt), session_seed(opt, 9100, i));
            cam_i.drift = Dr_i;
            [cam_i.Xp, cam_i.Yp] = meshgrid(cam_i.xs, cam_i.ys);
            GT_drift{i} = Dr_i;
            moves = true;
        end
        if use_jolts
            avoid = [];
            if use_impacts, avoid = I_i.mask; end
            J_i = make_jolts(F, jolt_options(opt), session_seed(opt, 9200, i), avoid);
            moves = moves || any(J_i.mask);
            % The baseline's low-pass part, fixed during jolts: a stand-in for
            % the vignetting, so none when the optics model it.
            jc = opt.jolt_cutoff;
            if isempty(jc)
                jc = 20;
                if use_optics && (op_p.vignetting || op_p.oof_layer), jc = 0; end
            end
            J_i.cutoff = jc;
            GT_jolts{i} = J_i;
            LBj0 = [];
            if jc > 0
                LBj0 = imgaussfilt(double(mult * BL(:, :, i)), jc);
            end
        end
        if use_optics
            % From streams of their own (seed + 8100 .. 8800 + session).
            op_s = op_p;
            if ~isempty(op_s.oof_layer_file) && opt.ses > 1
                [fp, fn, fe] = fileparts(op_s.oof_layer_file);
                op_s.oof_layer_file = fullfile(fp, sprintf('%s_ses%02d%s', fn, i - 1, fe));
            end
            op_i = make_camera_optics([d1, d2], F, op_s, opt.seed + i);
            if ~isempty(op_i.oof) && ~isempty(op_i.oof.save_file)
                if isfile(op_i.oof.save_file), delete(op_i.oof.save_file); end
                h5create(op_i.oof.save_file, '/L', [d1 d2 F], 'Datatype', 'double', ...
                    'ChunkSize', [d1 d2 1]);
            end
            cam_i.op = op_i;
            GT_optics{i} = rmfield(op_i, 'rs_hot');
            moves = moves || ~isempty(op_i.oof);
        end
    end

    % Session output setup
    if ~moves
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

    rs_noise = [];
    if opt.sensor_noise > 0
        rs_noise = RandStream('mt19937ar', 'Seed', opt.seed + 4000 + i);
    end
    G_i = [];
    if use_glitches
        G_i = make_glitches([d1, d2], F, glitch_options(opt), session_seed(opt, 9300, i));
        GT_glitches{i} = G_i;
        glitch_prev = [];
    end

    for start_idx = progress(1:batch_sz:F)
        batch_frames = start_idx:min(start_idx + batch_sz - 1, F);
        global_idx = session_frames(batch_frames);

        % --- Generate simulated session data in batches ---
        X_batch = reshape(reshape(A{i}, d1*d2, []) * C(:, global_idx), d1, d2, []) + ...
            reshape(reshape(bA{i}, d1*d2, []) * Cb(:, global_idx), d1, d2, []);
        if opt.smooth_background
            V_batch = to_grey(X_batch);
        else
            V_batch = v2uint8(X_batch);
        end
        if use_bursts
            scale_v = 2^8 / double(range(X_batch, 'all'));   % v2uint8's factor
        end
        clear X_batch

        % Add smoothed noise
        N_batch = rand(d1, d2, numel(batch_frames), 'single');
        N_batch = imgaussfilt(N_batch);
        N_batch = N_batch - mean(N_batch, 3);
        if opt.smooth_background
            N_batch = to_grey(N_batch);
        else
            N_batch = v2uint8(N_batch);
        end

        % Combine neural signal, noise, and baseline
        FV_batch = mult * BL(:, :, i) + ...
            (1 - mult) * V_batch * (1 / (1 / PNR + 1)) + ...
            (1 - mult) * N_batch * (1 - 1 / (1 / PNR + 1));
        if opt.scene_noise
            scale_a = 2^8 / double(range(FV_batch, 'all'));   % v2uint8's factor
            if opt.smooth_background
                FV_batch = to_grey(FV_batch);
            else
                FV_batch = v2uint8(FV_batch);
            end
        else
            % The same scene without its per-frame noise: the noise is still
            % drawn (the stream advances as always) but replaced by its mean,
            % and the frame keeps the grey scale it would have had with it.
            lo = double(min(FV_batch, [], 'all'));
            rg = double(range(FV_batch, 'all'));
            FV_batch = mult * BL(:, :, i) + ...
                (1 - mult) * V_batch * (1 / (1 / PNR + 1)) + ...
                (1 - mult) * mean(N_batch, 'all') * (1 - 1 / (1 / PNR + 1));
            if opt.smooth_background
                FV_batch = (double(FV_batch) - lo) / rg * 2^8;
            else
                FV_batch = uint8((double(FV_batch) - lo) / rg * 2^8);
            end
            scale_a = 2^8 / rg;
        end

        % Activity bursts, on this batch's grey scale: the neurons' and the
        % background's part of the scene is (1 - mult) * V_batch / (1/PNR + 1),
        % with V_batch = scale_v * (A C + bA Cb) less an offset.
        if use_bursts
            gain = scale_a * (1 - mult) * (1 / (1 / PNR + 1)) * scale_v;
            [FV_batch, GT_bursts{i}] = add_burst_image(FV_batch, GT_bursts{i}, ...
                A{i}, C_burst(:, global_idx), gain, batch_frames, opt.burst_glow_sigma);
        end

        if isempty(cam_i)
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
        else
            % All of it at once, through the camera model, with the sensor
            % noise added before the camera rounds each frame (see
            % apply_camera_motion for why the order matters).
            if use_impacts && any(I_i.mask(batch_frames))
                % The fixed and the foreign parts on this batch's grey scale
                % (the scene maps BL to it by the factor scale_a).
                cam_i.imp = struct('I', I_i, 'LB', scale_a * LB0, ...
                    'Hf_s', cam_i.My * (scale_a * mult * Hf0) * cam_i.Mx.');
            else
                cam_i.imp = [];
            end
            if use_jolts && any(J_i.mask(batch_frames))
                cam_i.jolt = struct('J', J_i, 'LB', scale_a * LBj0);
            else
                cam_i.jolt = [];
            end
            FV_batch = apply_camera_motion(FV_batch, batch_frames, cam_i, ...
                opt.sensor_noise, rs_noise);
        end

        % Sensor noise without the camera: after the last warp (each of which
        % rounds). White, independent per pixel and per frame, one frame at a
        % time from a stream of its own, so the draw does not depend on the
        % batch size and the motion-free twin gets the same one.
        if opt.sensor_noise > 0 && isempty(cam_i)
            for k = 1:size(FV_batch, 3)
                FV_batch(:, :, k) = uint8(double(FV_batch(:, :, k)) + ...
                    opt.sensor_noise * randn(rs_noise, d1, d2));
            end
        end

        % A smooth scene that nothing has rounded yet (no camera, no warp, no
        % sensor noise) is the still scene itself, written in floating point
        % (h5) or rounded (AVI); anything else is whole grey levels.
        if ~isa(FV_batch, 'uint8') && (opt.save_avi || opt.sensor_noise > 0)
            FV_batch = uint8(FV_batch);
        end

        % Acquisition glitches, on the frames as recorded.
        if ~isempty(G_i)
            [FV_batch, glitch_prev] = apply_glitches(FV_batch, batch_frames, G_i, glitch_prev);
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
            if use_bursts
                % The neurons' whole activity, bursts included.
                V_GT_batch = v2uint8(reshape(reshape(tA, d1*d2, []) * (C(:, global_idx) + C_burst(:, global_idx)), d1, d2, []) + ...
                    reshape(reshape(tbA, d1*d2, []) * Cb(:, global_idx), d1, d2, []));
            else
                V_GT_batch = v2uint8(reshape(reshape(tA, d1*d2, []) * C(:, global_idx), d1, d2, []) + ...
                    reshape(reshape(tbA, d1*d2, []) * Cb(:, global_idx), d1, d2, []));
            end
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
    clear tA tbA dx dy cam_i mo
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
    save([file_name '_meta.mat'], 'A_GT', 'A', 'C', 'S', 'BL', 'opt', 'Mot', 'LFP', 'GT_motion_xy', 'GT_nonrigid', 'GT_regimes', 'GT_impacts', 'placement', 'region_info', 'C_slow', 'Cb', 'Cb_slow', '-v7.3');
    if use_bursts
        save([file_name '_meta.mat'], 'GT_bursts', 'C_burst', '-append');
    end
end
if saveMat && (use_drift || use_jolts || use_glitches)
    save([file_name '_meta.mat'], 'GT_drift', 'GT_jolts', 'GT_glitches', '-append');
end

if saveMat && use_optics
    save([file_name '_meta.mat'], 'GT_optics', '-append');
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
% --- WITHIN-SESSION motion that changes over time (optional, off) ----------
% Drawn by make_motion_regimes from a stream of its own (seed + 3000 + session),
% so turning it on changes nothing else in the recording. Any of these, or
% camera_supersample > 1, switches every frame to the camera path: the shake,
% the bend and these are composed into one field and applied once.
% within_session_rigid_frames: [min max] length in frames of the rigid-only
%   stretches. [] (default) = no regimes, the bend is on throughout.
% within_session_bend_frames: [min max] length of the stretches with bends.
% within_session_regime_ramp: frames over which the bends fade in and out.
% within_session_local_bends: how many local bends are under way at once, on
%   average, inside a bend stretch. 0 (default) = none.
% within_session_local_bend_width: [min max] Gaussian sigma in pixels, drawn
%   log-uniformly.
% within_session_local_bend_amp: [min max] peak displacement in pixels.
% within_session_local_bend_life: [min max] lifetime in frames.
% within_session_local_bend_strain: cap on the peak strain (largest gradient)
%   of one bend, which caps the amplitude of the narrow ones.
% within_session_local_bend_inset: keep the bend centres this many pixels away
%   from the frame border (e.g. out of a margin that will be cropped).
% within_session_jumps: sudden rigid jumps per 1000 frames. 0 (default) = none.
% within_session_jump_size: [min max] length of a jump in pixels.
% within_session_jump_held: fraction of jumps that stay at the new place; the
%   others decay back exponentially.
% within_session_jump_decay: [min max] decay time constant in frames.
% within_session_jump_bound: no jump lands farther than this from the start.
% within_session_jump_gap: minimum frames between two jumps.
addParameter(inp, 'within_session_rigid_frames', [], @(x) isempty(x) || isnumeric(x))
addParameter(inp, 'within_session_bend_frames', [100 300], valid_v)
addParameter(inp, 'within_session_regime_ramp', 10, valid_v)
addParameter(inp, 'within_session_local_bends', 0, valid_v)
addParameter(inp, 'within_session_local_bend_width', [10 40], valid_v)
addParameter(inp, 'within_session_local_bend_amp', [0.5 3], valid_v)
addParameter(inp, 'within_session_local_bend_life', [40 200], valid_v)
addParameter(inp, 'within_session_local_bend_strain', 0.1, valid_v)
addParameter(inp, 'within_session_local_bend_inset', 0, valid_v)
addParameter(inp, 'within_session_jumps', 0, valid_v)
addParameter(inp, 'within_session_jump_size', [5 15], valid_v)
addParameter(inp, 'within_session_jump_held', 0.5, valid_v)
addParameter(inp, 'within_session_jump_decay', [10 40], valid_v)
addParameter(inp, 'within_session_jump_bound', 15, valid_v)
addParameter(inp, 'within_session_jump_gap', 100, valid_v)
% --- Slow drift (optional, off) ----------------------------------------------
% A session-long deformation that builds up at a rate (make_slow_drift): mostly
% rigid, with a small affine part (scale, rotation, shear) and small broad
% bends, slow in time, with a net displacement that does not average to zero.
% It is composed with the rest of the within-session motion (compose_slow_drift),
% not added. From a stream of its own (seed + 9100 + session). Ground truth in
% GT_drift. Takes the camera path. Each rate is drawn per session from
% [min max]; the ranges hold what was measured on real one-photon sessions,
% with room on both sides.
% slow_drift: true turns it on. false (default) = none.
% slow_drift_minutes: minutes of the drift's clock the movie spans. [] = the
%   movie's own length at 25 frames per second.
% slow_drift_rate: [min max] rigid speed, px/min, log-uniform.
% slow_drift_wander: [min max] how fast the headings wander, rad/sqrt(min).
% slow_drift_scale: [min max] %/min; slow_drift_rotation: [min max] deg/min;
%   slow_drift_shear: [min max] %/min; all log-uniform.
% slow_drift_bends: number of broad bends.
% slow_drift_bend_width: [min max] their Gaussian sigma, px, log-uniform.
% slow_drift_bend_rate: [min max] px/min, log-uniform.
% slow_drift_knot: minutes between the headings' knots.
addParameter(inp, 'slow_drift', false, valid_bool)
addParameter(inp, 'slow_drift_minutes', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x > 0))
addParameter(inp, 'slow_drift_rate', [0.02 0.5], valid_v)
addParameter(inp, 'slow_drift_wander', [0.2 0.8], valid_v)
addParameter(inp, 'slow_drift_scale', [0.005 0.15], valid_v)
addParameter(inp, 'slow_drift_rotation', [0.002 0.04], valid_v)
addParameter(inp, 'slow_drift_shear', [0.005 0.08], valid_v)
addParameter(inp, 'slow_drift_bends', 2, valid_v)
addParameter(inp, 'slow_drift_bend_width', [50 150], valid_v)
addParameter(inp, 'slow_drift_bend_rate', [0.005 0.08], valid_v)
addParameter(inp, 'slow_drift_knot', 0.5, valid_v)
% --- Jolts: impacts that keep the scene (optional, off) -----------------------
% A fast displacement of the SAME scene, as when the animal knocks the scope
% (make_jolts, jolt_frame): a few px, a fast rise, a slower return, with motion
% blur, a little defocus and a small brightness change; the baseline's
% low-pass part stays fixed to the optics. Beside the impacts below, whose view
% is lost to a foreign field of view (the rare stress case). From a stream of
% their own (seed + 9200 + session), kept jolt_gap frames away from the impacts.
% Ground truth in GT_jolts. Any jolt switches to the camera path.
% jolts: jolts per session at random onsets. 0 (default) = none.
% jolt_onset: their first frames instead ([] = random).
% jolt_size: [min max] peak displacement, px, uniform; the direction uniform.
% jolt_rise: [min max] frames to the peak.
% jolt_return: [min max] frames back.
% jolt_defocus: [min max] Gaussian sigma of the defocus at the peak, px.
% jolt_brightness: [min max] brightness change at the peak, a fraction; the
%   sign is random.
% jolt_cutoff: sigma in px of the Gaussian that splits the fixed low-pass part
%   off the baseline (as impact_cutoff), a stand-in for the vignetting. 0 = no
%   split, the whole frame moves. [] (default) = 20, or 0 when the optics'
%   vignetting or out-of-focus layer is on (they model what it stands in for).
% jolt_blur_step: px between the sub-frame instants of the motion blur.
% jolt_blur_max: at most this many instants per frame.
% jolt_gap: frames kept clear between jolts, impacts and the ends.
addParameter(inp, 'jolts', 0, valid_v)
addParameter(inp, 'jolt_onset', [], @(x) isempty(x) || isnumeric(x))
addParameter(inp, 'jolt_size', [1 8], valid_v)
addParameter(inp, 'jolt_rise', [1 6], valid_v)
addParameter(inp, 'jolt_return', [8 100], valid_v)
addParameter(inp, 'jolt_defocus', [0.5 2.5], valid_v)
addParameter(inp, 'jolt_brightness', [0.02 0.2], valid_v)
addParameter(inp, 'jolt_cutoff', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x >= 0))
addParameter(inp, 'jolt_blur_step', 0.25, valid_v)
addParameter(inp, 'jolt_blur_max', 32, valid_v)
addParameter(inp, 'jolt_gap', 50, valid_v)
% --- Acquisition glitches (optional, off) ---------------------------------------
% Dropped, white (saturated), black and partial frames, in short runs after
% which the next frame is normal (make_glitches, apply_glitches). Applied to
% the frames as recorded, after the motion, the camera and the sensor noise.
% From a stream of their own (seed + 9300 + session). Ground truth in
% GT_glitches.
% glitches: runs per 1000 frames. 0 (default) = none.
% glitch_run: [min max] frames per run.
% glitch_kinds: weights of dropped, white, black, partial.
% glitch_mixed: share of runs that draw a kind for every frame.
% glitch_white_gain: [min max] gain of a white frame, log-uniform;
%   glitch_white_offset: [min max] grey levels added to it.
% glitch_black_residual: largest gain left in a black frame;
%   glitch_black_level: [min max] its grey level.
% glitch_partial_band: [min max] share of the frame's height a partial frame
%   spoils; glitch_partial_cap: chance it is a cap from the top or the bottom
%   edge rather than a band; glitch_partial_white: chance the rows are white.
% glitch_inset: rows at each edge left out of that share (e.g. a margin that
%   will be cropped).
% glitch_gap: normal frames at least between two runs.
addParameter(inp, 'glitches', 0, valid_v)
addParameter(inp, 'glitch_run', [1 5], valid_v)
addParameter(inp, 'glitch_kinds', [1 1 1 1], valid_v)
addParameter(inp, 'glitch_mixed', 0.25, valid_v)
addParameter(inp, 'glitch_white_gain', [1.1 6], valid_v)
addParameter(inp, 'glitch_white_offset', [0 80], valid_v)
addParameter(inp, 'glitch_black_residual', 0.03, valid_v)
addParameter(inp, 'glitch_black_level', [0 3], valid_v)
addParameter(inp, 'glitch_partial_band', [0.1 0.7], valid_v)
addParameter(inp, 'glitch_partial_cap', 0.6, valid_v)
addParameter(inp, 'glitch_partial_white', 0.8, valid_v)
addParameter(inp, 'glitch_inset', 0, valid_v)
addParameter(inp, 'glitch_gap', 2, valid_v)
% --- Camera model (optional, off) ------------------------------------------
% camera_supersample: s > 1 records every frame as a camera would: each pixel
%   averages a band-limited version of the scene at s x s points, after the
%   within-session motion has moved it (camera_setup, apply_camera_motion).
%   The old path interpolates bilinearly, which pulls any correlation engine
%   about 0.1 px toward whole pixels; this does not. Odd s keeps the middle
%   sample on the pixel centre. 1 (default) = off.
addParameter(inp, 'camera_supersample', 1, valid_v)
% --- Camera optics and sensor (optional, off; make_camera_optics) ----------
% Fixed to the camera, applied after the motion: the light is multiplied by
% the vignetting, the dust and the brightness gain, then the sensor applies its
% gain texture and offsets (hot pixels, rows, columns, pixels), then the sensor
% noise. Every switch is off by default; the ranges below are drawn once per
% session (uniformly, sizes and the half-point log-uniformly) from streams of
% their own (seed + 8100 .. 8800 + session), and a range [a a] fixes a value.
% Grey levels are those of the recorded frame. Ranges are wider than the S4
% audit's measurements (PatchForge docs/lanes/s4-audit-m1-m4.md), which they
% contain. Any switch takes the camera path.
% camera_fov_inset: px on every side of the saved frame that are cropped off
%   later (the vignetting is fitted to the rest, and dust lands in it).
addParameter(inp, 'camera_fov_inset', 0, valid_v)
% hot_pixels: isolated pixels with an offset, keeping signal and noise.
%   hot_pixel_fraction [min max] of the pixels (log-uniform); offsets lognormal
%   with median hot_pixel_offset (grey levels) and log-sigma hot_pixel_spread;
%   with the sensor noise on, each has a noise of its own, std sqrt(c * offset),
%   c drawn from hot_pixel_noise.
addParameter(inp, 'hot_pixels', false, valid_bool)
addParameter(inp, 'hot_pixel_fraction', [2e-4 5e-3], valid_v)
addParameter(inp, 'hot_pixel_offset', [5 40], valid_v)
addParameter(inp, 'hot_pixel_spread', [0.4 1.1], valid_v)
addParameter(inp, 'hot_pixel_noise', [0 1], valid_v)
% sensor_pattern: the fixed pattern; stds of the per-pixel gain (fraction,
%   PRNU), the per-pixel offset (grey levels, DSNU), the row and column offsets.
addParameter(inp, 'sensor_pattern', false, valid_bool)
addParameter(inp, 'sensor_prnu', [0.005 0.04], valid_v)
addParameter(inp, 'sensor_dsnu', [0 1.5], valid_v)
addParameter(inp, 'sensor_row_offset', [0.1 1.5], valid_v)
addParameter(inp, 'sensor_col_offset', [0.1 1.5], valid_v)
% dust: static, multiplicative. dust_specks [min max] sharp dark specks (dust
%   on the sensor or filter), FWHM dust_size px, elongated up to 2:1, contrast
%   dust_contrast; dust_blurry [min max] faint soft disks (dust on a lens),
%   diameter dust_blurry_size px, contrast dust_blurry_contrast.
addParameter(inp, 'dust', false, valid_bool)
addParameter(inp, 'dust_specks', [0 6], valid_v)
addParameter(inp, 'dust_size', [2 8], valid_v)
addParameter(inp, 'dust_contrast', [0.05 0.5], valid_v)
addParameter(inp, 'dust_blurry', [0 3], valid_v)
addParameter(inp, 'dust_blurry_size', [20 120], valid_v)
addParameter(inp, 'dust_blurry_contrast', [0.02 0.15], valid_v)
% vignetting: multiplicative, from vignetting_file (variable V, peak 1; the
%   default was extracted from a real recording, see the file's 'source'),
%   scaled to cover the field of view; per session raised to a power drawn
%   from vignetting_strength, flipped up/down and left/right at random if
%   vignetting_flip, and moved by a fraction of the frame width drawn from
%   vignetting_shift in a random direction.
addParameter(inp, 'vignetting', false, valid_bool)
addParameter(inp, 'vignetting_file', 'vignetting_167-4.mat', @(x) ischar(x) || isstring(x))
addParameter(inp, 'vignetting_strength', [0.5 2], valid_v)
addParameter(inp, 'vignetting_flip', true, valid_bool)
addParameter(inp, 'vignetting_shift', [0 0.15], valid_v)
% oof_layer: out-of-focus light with a motion of its own (camera_layer_frame).
%   The split: a low-pass of shape oof_shape (one drawn per session from
%   'gaussian', 'dog', 'opening'), half-point oof_halfpoint px (Gaussian-sigma
%   equivalent, log-uniform), blur growing toward the corners by
%   oof_edge_factor. The motion: oof_gain times the tissue's field oof_lag
%   frames earlier, plus an AR(1) walk of std oof_walk px (smoothness
%   oof_walk_smoothness, capped at oof_walk_max). oof_layer_file: write every
%   frame's out-of-focus light there (h5, /L), for a replay proof.
addParameter(inp, 'oof_layer', false, valid_bool)
addParameter(inp, 'oof_shape', {'gaussian', 'dog', 'opening'}, @(x) ischar(x) || iscellstr(x) || isstring(x))
addParameter(inp, 'oof_halfpoint', [6 64], valid_v)
addParameter(inp, 'oof_edge_factor', [1 2], valid_v)
addParameter(inp, 'oof_gain', [0.5 1.2], valid_v)
addParameter(inp, 'oof_lag', [0 5], valid_v)
addParameter(inp, 'oof_walk', [0 2], valid_v)
addParameter(inp, 'oof_walk_smoothness', 0.99, valid_v)
addParameter(inp, 'oof_walk_max', 6, valid_v)
addParameter(inp, 'oof_layer_file', '', @(x) ischar(x) || isstring(x))
% flicker: a uniform gain 1 + g(t), g an AR(1) series of RMS flicker_rms and
%   smoothness flicker_smoothness, times 1 + sum of flicker_maps smooth maps
%   (Gaussian-filtered noise of scale flicker_map_scale px), each with its own
%   AR(1) series, together of RMS flicker_map_rms.
% brightness_trend: a slow exponential change of the gain, by trend_change at
%   the last frame (negative = dimmer, bleaching-like), time constant
%   trend_tau times the session's length.
addParameter(inp, 'flicker', false, valid_bool)
addParameter(inp, 'flicker_rms', [0.005 0.08], valid_v)
addParameter(inp, 'flicker_smoothness', [0 0.99], valid_v)
addParameter(inp, 'flicker_maps', 3, valid_v)
addParameter(inp, 'flicker_map_rms', [0 0.05], valid_v)
addParameter(inp, 'flicker_map_scale', [30 150], valid_v)
addParameter(inp, 'brightness_trend', false, valid_bool)
addParameter(inp, 'trend_change', [-0.3 0.05], valid_v)
addParameter(inp, 'trend_tau', [0.1 2], valid_v)
% black_level: the sensor's pedestal, grey levels added to every pixel's offset
%   before the noise and the rounding, so pixels with no light read it, stay
%   noisy and do not clip at 0. A range [a b] is drawn per session (uniform,
%   seed + 8700 + session). 0 (default) = none. The real control recordings
%   (S24): darkest mean pixel 6-10 grey, nothing clipped at 0.
addParameter(inp, 'black_level', 0, valid_v)
% adc_gain: the camera's digital gain g on its ADC's integer codes (missing
%   codes): the frame is round(g k + c), k = round((sensor + noise - c) / g),
%   so one 8-bit code in g / (g - 1) is never used, and where the noise is low
%   the smooth background's iso-grey contours every g / (g - 1) grey show as
%   terraces: concentric rings, fixed to the camera, after background removal.
%   Drawn per session (seed + 8800 + session): on with probability
%   adc_gain_prob, g uniform in the adc_gain range, c uniform in [0 g). 1
%   (default) = none. The real control recordings (S24): g 1.164 (one code in
%   7 missing).
%   adc_subpixels n: with the sensor noise on, each simulated pixel is read
%   by n x n sensor pixels (noise n * sensor_noise each, digitised to 8-bit
%   codes, the codes averaged),
%   as a recording binned n x n to the simulated pixel: the terraces are damped
%   by the sensor pixel's noise. 1 (default) = the simulated pixel is read once.
%   The real control recordings: n = 2 (S4: one simulated px ~ 2 x 2 sensor px).
addParameter(inp, 'adc_gain', 1, valid_v)
addParameter(inp, 'adc_gain_prob', 1, valid_v)
addParameter(inp, 'adc_subpixels', 1, valid_v)
% --- Sensor noise (optional, off) ------------------------------------------
% sensor_noise: standard deviation, in grey levels, of white Gaussian noise
%   added to every recorded frame after the motion, independent per pixel and
%   per frame, from a stream of its own (seed + 4000 + session): it does not
%   move with the tissue, and a motion-free twin gets the same draw. With the
%   camera model it is added before the camera rounds the frame, as a sensor
%   is read out and digitised once; without it, after the last warp. Rounded
%   and clipped to uint8. 0 (default) = none.
% scene_noise: false leaves the per-frame noise out of the scene (it is still
%   drawn, and replaced by its mean, so the grey scale and everything else stay
%   as they were). true (default) = as always: the noise is part of the scene,
%   so it moves with the tissue and the camera averages it. 'scene_noise',
%   false with 'sensor_noise', s moves the noise from the scene to the sensor.
addParameter(inp, 'sensor_noise', 0, valid_v)
addParameter(inp, 'scene_noise', true, valid_bool)
% --- Impacts (optional, off) -----------------------------------------------
% The animal knocks the microscope and the view is lost for about 1 s
% (make_impacts). The baseline is split by a Gaussian of impact_cutoff px into
% a low-pass part, fixed to the optics, and the rest of the scene (its
% high-pass part, the neurons, the background sources), which slides away fast
% with motion blur and blends into the high-pass part of another baseline
% image (impact_baseline); at the peak only that foreign structure shows. Then
% it reverses. Drawn from a stream of their own (seed + 5000 + session).
% Ground truth in GT_impacts. Any impact switches to the camera path.
% impacts: impacts per session at random onsets. 0 (default) = none.
% impact_onset: their first frames instead ([] = random).
% impact_duration: frames per impact (25, 1 s at the AVI writer's 25 fps).
% impact_distance: [min max] slide distance in px, drawn uniformly per impact;
%   the direction is uniform.
% impact_slide_frames: frames the slide takes, out and back.
% impact_blend_frames: frames the blend into the foreign structure takes, out
%   and back.
% impact_baseline: the baseline image (1-8, as 'B') the foreign structure comes
%   from.
% impact_cutoff: sigma in px of the Gaussian that splits low-pass from high-pass
%   (20: at 8 the broad vessels stay partly in the fixed low-pass part).
% impact_blur_samples: sub-frame instants averaged per frame (motion blur).
% impact_gap: frames kept clear between random impacts and from the ends.
addParameter(inp, 'impacts', 0, valid_v)
addParameter(inp, 'impact_onset', [], @(x) isempty(x) || isnumeric(x))
addParameter(inp, 'impact_duration', 25, valid_v)
addParameter(inp, 'impact_distance', [20 40], valid_v)
addParameter(inp, 'impact_slide_frames', 4, valid_v)
addParameter(inp, 'impact_blend_frames', 8, valid_v)
addParameter(inp, 'impact_baseline', 6, valid_v)
addParameter(inp, 'impact_cutoff', 20, valid_v)
addParameter(inp, 'impact_blur_samples', 16, valid_v)
addParameter(inp, 'impact_gap', 50, valid_v)
addParameter(inp, 'ses', 2, valid_v)           % Number of sessions to be generated
addParameter(inp, 'seed', 'shuffle')           % Random number generator seed ('shuffle' for a new seed each run)
addParameter(inp, 'B', '1')                    % Baseline id (1-8). Different valeus used different baseline images
addParameter(inp, 'spike_prob', [-4.9, 2.25])  % Probability distribution parameters for spike events (log scale mean and std)
addParameter(inp, 'save_videos', true, valid_bool)          % Flag to save session videos (true/false)
addParameter(inp, 'create_mask', true)         % Flag to create a mask during the simulation (true/false)
addParameter(inp, 'comb_fact', 0, valid_v)     % Combination factor used in the simulation process
addParameter(inp, 'drift', 0, valid_v)         % Presence of drifting activities in the simulated data (0 or 1)
addParameter(inp, 'slow_amp', 0, valid_v)      % Slow cyclic baseline: peak-to-trough, as a fraction of each neuron's peak (0 = none)
addParameter(inp, 'slow_period', [20 60], valid_v) % Its period in seconds, drawn uniformly per neuron from [min max]
addParameter(inp, 'slow_fraction', 1, valid_v) % Fraction of neurons that carry it
addParameter(inp, 'background', 'independent', @(x) any(strcmp(x, {'independent', 'per_cell'}))) % 'independent': n_background sources with their own traces; 'per_cell': one blurred copy of each cell, driven by it (the old behaviour)
addParameter(inp, 'n_background', 50, valid_v) % Number of independent background sources
addParameter(inp, 'slow_bg_amp', 0, valid_v)   % Slow cyclic baseline of the background sources, as slow_amp is for the neurons (0 = none)
% --- Smooth background (optional, off) -------------------------------------
% smooth_background: true keeps the scene (the baseline, the neurons, the
%   background sources, the scene noise) in floating point, on the grey scale
%   it always had, until the camera (or the warp, or the sensor noise) rounds
%   each frame once. false (default) = as always: the baseline is rounded to
%   8 bits before anything else, so every smooth contour of the scene is a
%   staircase of flat terraces that moves with the tissue. A still frame that
%   nothing rounds (no motion, no camera, no sensor noise) is written as it is,
%   in double (h5; AVI rounds it).
addParameter(inp, 'smooth_background', false, valid_bool)
% --- Activity bursts (optional, off) ---------------------------------------
% Periods of strong, synchronised, epileptic-like activity (make_bursts): many
% neurons fire together in rhythmic discharges for seconds, over the whole
% field or a patch of it, at once or as a wave spreading from a centre, and the
% diffuse background rises with them (a glow: the in-focus burst image
% blurred, the out-of-focus cells that burst too). The image changes, with no
% motion. Drawn from a stream of their own (seed + 11000 + session); added on
% the movie's own grey scale, so the frames outside bursts are what they would
% be without them (a very strong burst saturates at 255). Ground truth in
% GT_bursts and C_burst. Every [min max] option may have one row per burst
% instead (in time order), to give each burst its own range; Inf in a range
% means Inf.
% bursts: bursts per session at random onsets. 0 (default) = none.
% burst_onset: their first frames instead ([] = random).
% burst_duration: [min max] frames the population is driven, drawn per burst
%   (the calcium then decays for a few dozen frames more).
% burst_gap: frames kept clear between random bursts and from the ends.
% burst_ramp: frames the drive takes to rise and to fall.
% burst_fraction: [min max] probability that a neuron at the burst's centre is
%   recruited, drawn per burst.
% burst_radius: [min max] px, drawn per burst: recruitment falls off as
%   exp(-d^2 / (2 radius^2)) with the distance d from the burst's centre
%   (uniform in the frame). Inf (default) = the whole field alike.
% burst_speed: [min max] px per frame, drawn per burst: each neuron fires
%   d / speed frames after the population (a wave from the centre). Inf
%   (default) = all at once.
% burst_amplitude: [min max] peak of a recruited neuron's burst trace, in units
%   of its own largest transient in the session; drawn per neuron and burst.
% burst_neuropil: [min max] the glow's total light over the in-focus burst
%   image's, drawn per burst. 0 = no glow.
% burst_glow_sigma: px, the Gaussian blur of the glow.
% burst_interval: [min max] frames between the population's discharges.
% burst_join: [min max] probability that a recruited neuron joins a discharge
%   at full drive, drawn per neuron and burst.
% burst_tail: a burst ends where its population trace has decayed below this
%   fraction of its peak; it adds nothing after.
addParameter(inp, 'bursts', 0, valid_v)
addParameter(inp, 'burst_onset', [], @(x) isempty(x) || isnumeric(x))
addParameter(inp, 'burst_duration', [25 100], valid_v)
addParameter(inp, 'burst_gap', 50, valid_v)
addParameter(inp, 'burst_ramp', 5, valid_v)
addParameter(inp, 'burst_fraction', [0.4 0.8], valid_v)
addParameter(inp, 'burst_radius', [Inf Inf], valid_v)
addParameter(inp, 'burst_speed', [Inf Inf], valid_v)
addParameter(inp, 'burst_amplitude', [1 2], valid_v)
addParameter(inp, 'burst_neuropil', [1 2], valid_v)
addParameter(inp, 'burst_glow_sigma', 20, valid_v)
addParameter(inp, 'burst_interval', [2 6], valid_v)
addParameter(inp, 'burst_join', [0.5 1], valid_v)
addParameter(inp, 'burst_tail', 0.005, valid_v)
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

function [C, C_slow] = add_slow_baseline(C, opt, slow_amp, seed_offset)
% A slow cyclic baseline per trace: a raised cosine from 0 to slow_amp times
% the trace's peak, with its own period (slow_period, seconds) and phase. It is
% added to the trace and returned on its own (C_slow) so what is recovered can
% be scored. It draws from a stream of its own (seed + seed_offset), so the rest
% of the recording is the same as without it.
C_slow = zeros(size(C), 'like', C);
if slow_amp <= 0
    return
end
if isnumeric(opt.seed)
    rs = RandStream('mt19937ar', 'Seed', opt.seed + seed_offset);
else
    rs = RandStream('mt19937ar', 'Seed', 'shuffle');
end
N = size(C, 1);
t = (0:size(C, 2) - 1) / opt.sf;                       % seconds
period = opt.slow_period(1) + rand(rs, N, 1) * diff(opt.slow_period([1 end]));
phase = 2*pi*rand(rs, N, 1);
has = rand(rs, N, 1) < opt.slow_fraction;
amp = slow_amp * max(C, [], 2) .* has;
C_slow = cast(amp .* (1 - cos(2*pi*t ./ period + phase)) / 2, 'like', C);
C = C + C_slow;
end

function out = to_grey(in)
% v2uint8 without the rounding: the same grey scale, in double.
in = double(in);
out = (in - min(in, [], 'all')) ./ range(in, 'all') * 2^8;
end

function [use, GT_bursts, C_burst] = session_bursts(C, placement, opt)
% Activity bursts for every session (make_bursts), or none.
use = opt.bursts > 0 || ~isempty(opt.burst_onset);
GT_bursts = cell(1, opt.ses);
C_burst = [];
if ~use
    return
end
% The simulator's kinetics, as create_neuron_data scales them to sf.
rise = opt.invtRise; rise(1) = log(exp(rise(1)) / opt.sf);
decay = opt.invtDecay; decay(1) = log(exp(decay(1)) / opt.sf);
p = struct('n', opt.bursts, 'onset', opt.burst_onset, 'duration', opt.burst_duration, ...
    'gap', opt.burst_gap, 'ramp', opt.burst_ramp, 'fraction', opt.burst_fraction, ...
    'radius', opt.burst_radius, 'speed', opt.burst_speed, ...
    'amplitude', opt.burst_amplitude, 'neuropil', opt.burst_neuropil, ...
    'interval', opt.burst_interval, 'join', opt.burst_join, 'tail', opt.burst_tail, ...
    'rise', rise, 'decay', decay);
xy = [placement.X_saved(:), placement.Y_saved(:)];
saved_size = opt.d - 40;                 % Add_NRmotion crops 20 px per side
C_burst = zeros(size(C));
F = opt.F;
for i = 1:opt.ses
    cols = (i-1)*F+1:i*F;
    Bu = make_bursts(C(:, cols), xy, saved_size, p, opt.seed + 11000 + i);
    C_burst(:, cols) = Bu.C_burst;
    Bu = rmfield(Bu, 'C_burst');
    % Filled in frame by frame as the image is made (add_burst_image).
    Bu.added_mean = zeros(1, F);     % mean grey level the bursts add, saved frame
    Bu.added_max = zeros(1, F);      % largest grey level they add
    Bu.peak_image = [];              % [d1 d2 n] what they add at each peak frame
    GT_bursts{i} = Bu;
end
end

function [FV, Bu] = add_burst_image(FV, Bu, A, Cn, gain, frames, glow_sigma)
% Add the bursts' image to the frames FV (on their grey scale, uint8 or
% double): the in-focus part, gain * A Cn, plus its glow, the same blurred by
% a Gaussian of glow_sigma px and scaled by the burst's glow ratio (total
% light over the in-focus part's). Record what was added (Bu.added_*,
% Bu.peak_image). Only the frames with burst activity are touched.
[d1, d2, ~] = size(FV);
on = find(any(Cn, 1));
if isempty(Bu.peak_image)
    Bu.peak_image = zeros(d1, d2, size(Bu.table, 1), 'single');
end
for k = on
    t = frames(k);
    add = gain * reshape(reshape(A, d1*d2, []) * Cn(:, k), d1, d2);
    if Bu.glow_ratio(t) > 0
        add = add + Bu.glow_ratio(t) * imgaussfilt(add, glow_sigma, 'Padding', 0);
    end
    if isa(FV, 'uint8')
        FV(:, :, k) = uint8(double(FV(:, :, k)) + add);
    else
        FV(:, :, k) = FV(:, :, k) + add;
    end
    Bu.added_mean(t) = mean(add, 'all');
    Bu.added_max(t) = max(add, [], 'all');
    j = find(Bu.table(:, 6) == t);
    if ~isempty(j)
        Bu.peak_image(:, :, j) = single(add);
    end
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

function p = optics_options(opt)
% The options make_camera_optics reads (camera_fov_inset as fov_inset).
names = {'hot_pixels', 'hot_pixel_fraction', 'hot_pixel_offset', 'hot_pixel_spread', ...
    'hot_pixel_noise', 'sensor_pattern', 'sensor_prnu', 'sensor_dsnu', ...
    'sensor_row_offset', 'sensor_col_offset', 'dust', 'dust_specks', 'dust_size', ...
    'dust_contrast', 'dust_blurry', 'dust_blurry_size', 'dust_blurry_contrast', ...
    'vignetting', 'vignetting_file', 'vignetting_strength', 'vignetting_flip', ...
    'vignetting_shift', 'oof_layer', 'oof_shape', 'oof_halfpoint', 'oof_edge_factor', ...
    'oof_gain', 'oof_lag', 'oof_walk', 'oof_walk_smoothness', 'oof_walk_max', ...
    'oof_layer_file', 'flicker', 'flicker_rms', 'flicker_smoothness', 'flicker_maps', ...
    'flicker_map_rms', 'flicker_map_scale', 'brightness_trend', 'trend_change', 'trend_tau', ...
    'black_level', 'adc_gain', 'adc_gain_prob', 'adc_subpixels'};
p = struct('fov_inset', opt.camera_fov_inset);
for k = 1:numel(names)
    p.(names{k}) = opt.(names{k});
end
for k = {'hot_pixels', 'sensor_pattern', 'dust', 'vignetting', 'oof_layer', ...
        'flicker', 'brightness_trend', 'vignetting_flip'}
    p.(k{1}) = logical(p.(k{1}));
end
p.vignetting_file = char(p.vignetting_file);
p.oof_layer_file = char(p.oof_layer_file);
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
function p = regime_options(opt)
% The within_session_* options make_motion_regimes reads, without the prefix.
names = {'rigid_frames', 'bend_frames', 'regime_ramp', 'local_bends', ...
    'local_bend_width', 'local_bend_amp', 'local_bend_life', 'local_bend_strain', ...
    'local_bend_inset', 'jumps', 'jump_size', 'jump_held', 'jump_decay', ...
    'jump_bound', 'jump_gap'};
p = struct();
for k = 1:numel(names)
    p.(names{k}) = opt.(['within_session_' names{k}]);
end
end

function p = impact_options(opt)
% The impact_* options make_impacts reads, without the prefix.
p = struct('n', opt.impacts, 'onset', opt.impact_onset, ...
    'duration', opt.impact_duration, 'distance', opt.impact_distance, ...
    'slide_frames', opt.impact_slide_frames, 'blend_frames', opt.impact_blend_frames, ...
    'blur_samples', opt.impact_blur_samples, 'gap', opt.impact_gap);
end

function Hf = foreign_structure(opt, saved_size)
% The high-pass part of baseline image opt.impact_baseline, made as the
% scene's own baseline is (create_baseline, the 20 px Add_NRmotion crops,
% v2uint8), on its grey scale. The global random state is left untouched.
prev = rng;
o = opt; o.B = opt.impact_baseline; o.ses = 1;
bl = create_baseline(o);
rng(prev);
bl = bl(21:20 + saved_size(1), 21:20 + saved_size(2), 1);
if opt.smooth_background
    bl = to_grey(bl);
else
    bl = double(v2uint8(bl));
end
Hf = bl - imgaussfilt(bl, opt.impact_cutoff);
end

function p = slow_drift_options(opt)
% The slow_drift_* options make_slow_drift reads, without the prefix.
names = {'minutes', 'rate', 'wander', 'scale', 'rotation', 'shear', 'bends', ...
    'bend_width', 'bend_rate', 'knot'};
p = struct();
for k = 1:numel(names)
    p.(names{k}) = opt.(['slow_drift_' names{k}]);
end
end

function p = jolt_options(opt)
% The jolt_* options make_jolts reads, without the prefix.
p = struct('n', opt.jolts, 'onset', opt.jolt_onset, 'size', opt.jolt_size, ...
    'rise', opt.jolt_rise, 'return_frames', opt.jolt_return, ...
    'defocus', opt.jolt_defocus, 'brightness', opt.jolt_brightness, ...
    'blur_step', opt.jolt_blur_step, 'blur_max', opt.jolt_blur_max, 'gap', opt.jolt_gap);
end

function p = glitch_options(opt)
% The glitch_* options make_glitches reads, without the prefix.
names = {'run', 'kinds', 'mixed', 'white_gain', 'white_offset', 'black_residual', ...
    'black_level', 'partial_band', 'partial_cap', 'partial_white', 'inset', 'gap'};
p = struct('rate', opt.glitches);
for k = 1:numel(names)
    p.(names{k}) = opt.(['glitch_' names{k}]);
end
end

function s = session_seed(opt, offset, i)
% The seed of a per-session stream, as the motion streams above take theirs.
if isempty(opt.within_session_motion_seed)
    s = opt.seed + offset + i;
else
    s = opt.within_session_motion_seed + offset + i - 1;
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
