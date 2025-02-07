function save_as_avi(V,videoFileName)
% Create a VideoWriter object

v = VideoWriter(videoFileName, 'Uncompressed AVI');  % Use 'Motion JPEG AVI' for compression
v.FrameRate = 25;  % Set frame rate (frames per second)

% Open the video file
open(v);

% Write frames to the video
for k = 1:size(V,3)
    frame = V(:, :, k);  % Extract the k-th frame
    writeVideo(v, frame);      % Write the frame to the video
end

% Close the video file
close(v);
