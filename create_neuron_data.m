function [Aout,bA,C,S,LFP]=create_neuron_data(opt)
%% Create Mask
if opt.create_mask
    BW=create_heart_mask(opt.d);
else
    BW=ones(opt.d);
end
%% Create random spatial
[Aout,bA]=create_spatial(BW,opt);
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
for i=1:opt.ses
    [s,lfp]=create_random_spike(N,opt.F,opt.sf,P,opt.LFP,1,opt.force_active);
    LFP=[LFP,lfp];
    S=[S,s];
end
S=fix_inactive(S);
C=conv_spikes(S,invtRise,invtDecay,size(S,2));


% C=C.*(1+randn(length(X),1).*0.2);
%

end


function [Aout,bA]=create_spatial(BW,opt)
[y, x] = find(BW);
[X,Y]=get_random_points(x,y,opt.min_dist,opt.Nneu,opt.plotme);
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





