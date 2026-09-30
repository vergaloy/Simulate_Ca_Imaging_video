function [dM,dA,dbA,D]=Add_NRmotion(M,A,bA,across_session_nonrigid,plotme,across_session_translation,across_session_nonrigid_scale)
%% Add_NRmotion: misalign one session against the reference session.
%
% ACROSS-SESSION only. One warp, applied to the baseline and to every neuron
% footprint of this session, so the whole recording sits in a shifted and bent
% coordinate frame. It does not vary from frame to frame; the frame-to-frame
% motion is applied later, inside Simulate_Ca_video.
%
% Inputs:
%   M, A, bA                      baseline, neuron and blood-vessel footprints
%   across_session_nonrigid       amplitude of the smooth warp, in pixels
%   plotme                        draw the flow field and the warped frame
%   across_session_translation    length of the rigid shift, in pixels. The
%                                 direction is random, the length is this.
%   across_session_nonrigid_scale Gaussian sigma in pixels used to smooth the
%                                 random field, i.e. how broad the warp is
%
% Outputs:
%   dM, dA, dbA  the warped inputs, cropped 20 px on every border
%   D            the displacement field that was applied, same crop
if ~exist('across_session_nonrigid','var')
    across_session_nonrigid=2;
end

if ~exist('plotme','var')
    plotme=0;
end

if ~exist('across_session_translation','var')
    across_session_translation=0;
end

if ~exist('across_session_nonrigid_scale','var')
    across_session_nonrigid_scale=60;
end

[d1,d2]=size(M);
img1=mat2gray(imgaussfilt(randn(d1,d2),across_session_nonrigid_scale,'FilterSize',2001,'Padding','circular'));
[Gx,Gy] = imgradientxy(img1);
if across_session_translation>0
    T=randn(1,2);
    T=T*(across_session_translation/sqrt(sum(T.^2)));
    Gx=(mat2gray(Gx)-0.5)*2*across_session_nonrigid+T(1);
    Gy=(mat2gray(Gy)-0.5)*2*across_session_nonrigid+T(2);
else
    Gx=(mat2gray(Gx)-0.5)*2*across_session_nonrigid;
    Gy=(mat2gray(Gy)-0.5)*2*across_session_nonrigid;
end
D=[];
D(:,:,1)=Gy;
D(:,:,2)=Gx;

dM = imwarp(M,D);
dM =dM (21:d1-20,21:d2-20,:);

dA = imwarp(A,D);
dA =dA(21:d1-20,21:d2-20,:);

dbA = imwarp(bA,D);
dbA =dbA(21:d1-20,21:d2-20,:);

M =M (21:d1-20,21:d2-20,:);
D=D(21:d1-20,21:d2-20,:);

if (plotme==1)
    subplot(2,2,1);
%     imshow(zeros(d1,d2));hold on;
    imshow(M);hold on;
    flow = opticalFlow(-Gx,-Gy);
    h1=plot(flow,'DecimationFactor',[25 25]); hold off;
    h = get(gca, 'Children');
    set(h(1), 'Color', 'b');
    
    subplot(2,2,2);imshow(dM, []); 
    subplot(2,2,3);imshow(mat2gray(M-dM));
    
    Vf1=vesselness_PV(M,0);
    Vf2=vesselness_PV(dM,0);
    If= imfuse(Vf1, Vf2,'ColorChannels',[1 2 0]);subplot(2,2,4);imshow(If)    
end

end
