function [out,Mot,outpath,file_name,A]=Simulate_Ca_video(varargin)
% Simulate_Ca_video('motion',0,'Nneu',100,'ses',2,'CA1_A',true,'PNR',1);
% Simulate_Ca_video('Nneu',50,'ses',1,'F',18000,'LFP',8,'spike_prob',[-4.91,0.83],'sf',60);
% Simulate_Ca_video('Nneu',300,'ses',20,'F',1500,'motion',0,'min_dist',2,'spike_prob',[-2,0.83],'A2',true,'overlap',0.2);
% Simulate_Ca_video_main(Inputs);
% Simulate_Ca_video('Nneu',100,'ses',4,'F',500,'save_files',true,'motion',2,'translation',15,'overlap',0.5);

% V=Simulate_Ca_video('save_files',false,raw_data{1,:});
opt=int_var(varargin);
outpath=get_out_path(opt);

%% Create Baselines for two sessions;
bl=create_baseline(opt);
%% create neural data
[A_GT,bA_GT,C,S,LFP]=create_neuron_data(opt);
opt.Nneu2=size(A_GT{1,1},3);
%% Apply overlapping mask tco C
C=add_remapping_drifting(C,opt);

%% add Non-Rigid motion to sessions
[BL(:,:,1),A{1,1},bA{1,1},Mot{1}]=Add_NRmotion(bl(:,:,1),A_GT{1},bA_GT{1},0,opt.plotme,opt.translation,opt.motion_sz);
for i=2:opt.ses
    [BL(:,:,i),A{1,i},bA{1,i},Mot{i}]=Add_NRmotion(bl(:,:,i),A_GT{i},bA_GT{i},opt.motion,opt.plotme,opt.translation,opt.motion_sz);
end
BL=v2uint8(BL);
%% create noise data
[d1,d2]=size(BL(:,:,1));
N=rand(d1,d2,opt.F*opt.ses,'single');
N=imgaussfilt(N);
N=N-mean(N,3);

N=v2uint8(N);
%% Integrate model
out=cell(1,opt.ses);
GT=cell(1,opt.ses);
ix1=1;
F=opt.F;
ix2=F;
PNR=opt.PNR;
mult=0.8;  %% BL proportion to total signal.
for i=1:opt.ses
    V=v2uint8(reshape(reshape(A{i},d1*d2,[])*C(:,ix1:ix2),d1,d2,[])+...
        reshape(reshape(bA{i},d1*d2,[])*C(:,ix1:ix2),d1,d2,[]));
    tn=N(:,:,ix1:ix2);
    FV=mult*BL(:,:,i)+  (1-mult)*(V)*(1/(1/PNR+1))   +   (1-mult)*tn*(1-1/(1/PNR+1)); %add data
    FV=v2uint8(FV);  %
    out{1,i}=FV;
    tA=A_GT{i};
    tA =tA(21:size(tA,1)-20,21:size(tA,2)-20,:);
    tbA=bA_GT{i};
    tbA =tbA(21:size(tbA,1)-20,21:size(tbA,2)-20,:);
    V=v2uint8(reshape(reshape(tA,d1*d2,[])*C(:,ix1:ix2),d1,d2,[])+reshape(reshape(tbA,d1*d2,[])*C(:,ix1:ix2),d1,d2,[]));
    GT{1,i}=V*(1/(1/PNR+1))+tn.*(1-1/(1/PNR+1));
    ix1=ix1+F;
    ix2=ix2+F;
end

GT=cat(3,GT{:});
%% Save results
if opt.save_files

    file_name=datestr(now,'yymmdd_HHMMSS');
    for i=1:opt.ses
        if opt.save_avi
            save_as_avi(out{1,i}+1,[file_name,'_ses',sprintf( '%02d',i-1),'_mc.avi']);
        else
            saveash5(out{1,i}+1,[file_name,'_ses',sprintf( '%02d',i-1),'_mc']);
        end
    end
    if opt.save_avi
        save_as_avi(GT+1,[file_name,'_GT.avi'])
    else

        saveash5(GT+1,[file_name,'_GT']);
    end
    save([file_name,'.mat'],'A_GT','A','C','S','BL','opt','Mot','LFP','-v7.3');
else
    file_name=[];
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
addParameter(inp, 'motion', 0, valid_v)        % Inter-session missaligment amplitude (0 for none)
addParameter(inp, 'motion_sz', 60, valid_v)    % Size (magnitude) of the motion effect applied to the frames
addParameter(inp, 'translation', 1, valid_v)   % Add translation missaligment in addition to Non-rigid
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



