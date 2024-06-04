function [S,LFP]=create_random_spike(N,F,sf,P,lfp,lfp_strength,force_active)
% out=create_random_spike(100,6000);
if ~exist('lfp_strength','var')
    lfp_strength=1;
end


if lfp>0
    P=P*2;
    P(P>1)=1;
    [LFP,LFPp]=create_sine(lfp,sf,F);
else
    LFPp=ones(1,F);
    LFP=ones(1,F);
end

%% create spikes
for i=1:N
    p=P(i);
    bin=size(LFPp,2)/F;
    p=p/bin;
    p=p.*LFPp;
    temp=double((rand(size(p))-p)<0);
    if force_active==1
        if sum(temp)==0
            ix=1:numel(p);
            ix=ix(p>0);
            pnt=ix(randi(numel(ix)));
            temp(pnt)=1;
        end
    end
    S(i,:)=temp;
end

if lfp_strength<1  %% shuffle some spike
    ds=datasample(1:N,floor((1-lfp_strength)*N),'Replace',false);
    for i=1:numel(ds)
        S(ds(i),:)=S(ds(i),randperm(size(S,2),size(S,2)));
    end
end


S=bin_data(S,1,bin);


end


function [LFP,fP]=create_sine(lfp,sf,F)
lfp_sf=120;
Freq=lfp-4:0.01:lfp+5;
LFP=zeros(1,(F/sf)*lfp_sf);
dt = 1/lfp_sf; % seconds per sample
StopTime = (F/lfp_sf); % seconds
t = (dt:dt:StopTime*(lfp_sf/sf)); % seconds
for i=1:numel(Freq)
    LFP_t= sin(2.*pi.*Freq(i).*t+randn(1)*3);
    LFP=LFP+LFP_t*normpdf(Freq(i),lfp,1);
end

% [pxx ,f]= pwelch(LFP,500,250,500,120);
% plot(f,pxx)

[yupper,ylower] = envelope(LFP);
fP=(LFP-ylower)./(yupper-ylower);

% fP = resample(fP,sf,lfp_sf);

end


