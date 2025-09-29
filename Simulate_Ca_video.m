function [Mot, outpath, file_name, A, GT_motion_xy] = Simulate_Ca_video(varargin)
% Simulate_Ca_video generates synthetic calcium imaging videos with optional across-session misalignment, within-session motion, and remapping.
% It simulates neural activity, misalignment artifacts, baseline variability, and saves session-wise videos and metadata.

% Example usages:
% Simulate_Ca_video('NR_misalignment',0,'Nneu',100,'ses',2,'CA1_A',true,'PNR',1);
% Simulate_Ca_video('Nneu',300,'ses',20,'F',1500,...);
% V = Simulate_Ca_video('save_files',false,raw_data{1,:});

% Parse inputs and initialize options
opt = int_var(varargin);
outpath = get_out_path(opt);  % choose output folder if saving enabled

%% Generate synthetic data
bl = create_baseline(opt);  % create session-specific background baselines
[A_GT, bA_GT, C, S, LFP] = create_neuron_data(opt);  % simulate neurons and temporal traces
opt.Nneu2 = size(A_GT{1,1}, 3);  % store number of neurons

C = add_remapping_drifting(C, opt);  % apply remapping or activity drift if needed

% Apply synthetic non-rigid misalignment to neuron components for each session
[BL(:,:,1), A{1,1}, bA{1,1}, Mot{1}] = Add_NRmotion(bl(:,:,1), A_GT{1}, bA_GT{1}, ...
    0, opt.plotme, opt.translation_misalignment, opt.motion_sz);
for i = 2:opt.ses
    [BL(:,:,i), A{1,i}, bA{1,i}, Mot{i}] = Add_NRmotion(bl(:,:,i), A_GT{i}, bA_GT{i}, ...
        opt.NR_misalignment, opt.plotme, opt.translation_misalignment, opt.motion_sz);
end
BL = v2uint8(BL);  % convert baseline to uint8

[d1, d2] = size(BL(:,:,1));  % get spatial dimensions

GT_motion_xy = cell(1, opt.ses);  % store within-session motion ground truth

%% Simulate each session one-by-one (memory-efficient)
file_name = datestr(now, 'yymmdd_HHMMSS');  % timestamp-based base filename
ix1 = 1; F = opt.F; ix2 = F;
PNR = opt.PNR; mult = 0.8;

if opt.save_files
    % Prepare memory-mapped file for writing GT frames incrementally
    matGT = matfile([file_name '_GT.mat'], 'Writable', true);
end

for i = 1:opt.ses
    % --- Generate simulated session data ---
    V = v2uint8(reshape(reshape(A{i}, d1*d2, []) * C(:, ix1:ix2), d1, d2, []) + ...
        reshape(reshape(bA{i}, d1*d2, []) * C(:, ix1:ix2), d1, d2, []));

    % Add smoothed noise
    N = rand(d1, d2, F, 'single');
    N = imgaussfilt(N);
    N = N - mean(N, 3);
    N = v2uint8(N);

    % Combine neural signal, noise, and baseline
    FV = mult * BL(:, :, i) + ...
        (1 - mult) * V * (1 / (1 / PNR + 1)) + ...
        (1 - mult) * N * (1 - 1 / (1 / PNR + 1));
    FV = v2uint8(FV);

    % Apply within-session translational motion if requested
    if opt.session_motion_std > 0
        prev_rng = rng;
        if isempty(opt.session_motion_seed)
            seed_i = opt.seed + i;
        else
            seed_i = opt.session_motion_seed + i - 1;
        end
        [dx, dy] = make_xy_motion_ar(F, opt.session_motion_phi, opt.session_motion_std, ...
            opt.session_motion_max, seed_i);
        rng(prev_rng);
        FV = apply_xy_translation(single(FV), dx, dy);
        FV = v2uint8(FV);
        GT_motion_xy{i} = {dx, dy};
    else
        GT_motion_xy{i} = [];
    end

    % --- Save session output video ---
    if opt.session_motion_std==0
        mc_tag='_mc';
    else
        mc_tag='';
    end
    if opt.save_avi
        save_as_avi(FV + 1, [file_name '_ses' sprintf('%02d', i-1) mc_tag '.avi']);
    else
        saveash5(FV + 1, [file_name '_ses' sprintf('%02d', i-1) mc_tag]);
    end

    % --- Ground truth generation ---
    tA = A_GT{i}(21:end-20, 21:end-20, :);     % crop edges
    tbA = bA_GT{i}(21:end-20, 21:end-20, :);
    V_GT = v2uint8(reshape(reshape(tA, d1*d2, []) * C(:, ix1:ix2), d1, d2, []) + ...
        reshape(reshape(tbA, d1*d2, []) * C(:, ix1:ix2), d1, d2, []));

    % Write GT session to disk incrementally
    if opt.save_files
        if i == 1
            matGT.GT = zeros(d1, d2, F * opt.ses, 'uint8');  % preallocate on disk
        end
        matGT.GT(:, :, ix1:ix2) = V_GT;  % memory-safe write
    end

    % Update frame index
    ix1 = ix1 + F;
    ix2 = ix2 + F;

    % Clean session variables from memory
    clear V N FV V_GT tA tbA
end

%% Final saving
if opt.save_files
    % Save full GT video
    if opt.save_avi
        save_as_avi(matGT.GT + 1, [file_name '_GT.avi']);
    else
        saveash5(matGT.GT + 1, [file_name '_GT']);
    end

    % Save simulation metadata (everything else)
    save([file_name '_meta.mat'], 'A_GT', 'A', 'C', 'S', 'BL', 'opt', 'Mot', 'LFP', 'GT_motion_xy', '-v7.3');
else
    file_name = [];
end

end
function opt=int_var(varargin)
%% INTIALIZE VARIABLES
inp = inputParser;
valid_v = @(x) isnumeric(x);
addParameter(inp, 'outpath', [])               % Output path where simulation results will be saved
addParameter(inp, 'min_dist', 8, valid_v)      % Minimum distance between neurons in the simulated data
addParameter(inp, 'Nneu', 50, valid_v)         % Number of neurons (or neural units) to be simulated
addParameter(inp, 'PNR', 2, valid_v)          % Peak-to-Noise Ratio (signal-to-noise ratio) for the simulated data
addParameter(inp, 'd', [220, 300], valid_v)    % Dimensions of the simulated video frames (width and height) in pixels
addParameter(inp, 'F', 1000, valid_v)          % Frame rate (frames per second) for the simulated calcium video data in each session
addParameter(inp, 'overlap', 1, valid_v)       % Proportion of neurons remapping across multiple sessions
addParameter(inp, 'overlapMulti', 0, valid_v)  % Remapping strenght. 0 means 100% remapping (should be the opposite, sorry!)
addParameter(inp, 'NR_misalignment', 0, valid_v)        % Inter-session non-rigid misalignment amplitude (0 for none)
addParameter(inp, 'motion_sz', 60, valid_v)    % Size (magnitude) of the misalignment effect applied to the frames
addParameter(inp, 'translation_misalignment', 1, valid_v)   % Add translation misalignment in addition to non-rigid component
addParameter(inp, 'session_motion_std', 3, valid_v) % Within-session translational motion std (pixels)
addParameter(inp, 'session_motion_phi', 0.99, valid_v) % Temporal smoothness for within-session motion (AR coefficient)
addParameter(inp, 'session_motion_max', [], @(x) isnumeric(x) || isempty(x)) % Optional peak cap for within-session motion
addParameter(inp, 'session_motion_seed', [], valid_v) % Seed offset for reproducible within-session motion
addParameter(inp, 'ses', 2, valid_v)           % Number of sessions to be generated
addParameter(inp, 'seed', 'shuffle')           % Random number generator seed ('shuffle' for a new seed each run)
addParameter(inp, 'B', '1')                    % Baseline id (1-8). Different valeus used different baseline images
addParameter(inp, 'spike_prob', [-4.9, 2.25])  % Probability distribution parameters for spike events (log scale mean and std)
addParameter(inp, 'save_files', true)          % Flag to save simulation results (true/false)
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
addParameter(inp, 'A2', false)                  % Use CA1 neurons shapes instead of DG.
addParameter(inp, 'force_active', 1)           % All nuerons should have at least one Calcium transient per session (unless remapping)
addParameter(inp, 'save_avi', 1)           % Save as avi. otherwise save as .h5
varargin=varargin{1, 1};

inp.KeepUnmatched = true;
parse(inp,varargin{:});
opt=inp.Results;
opt.B=str2num(string(opt.B));


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
if opt.save_files
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
