function [FV, prev] = apply_glitches(FV, frames, G, prev)
%% Spoil the recorded frames that make_glitches chose.
%
% FV is a batch of recorded uint8 frames (after the motion, the camera and the
% sensor noise), frames their indices within the session, prev the last frame
% of the previous batch as written ([] at the start). Dropped frames repeat the
% frame written before them; white, black and partial frames are
% uint8(gain * frame + offset) on their spoiled rows. Returns the batch and its
% last frame as written, for the next batch.
%
% Author: Pablo Vergara

for k = 1:size(FV, 3)
    t = frames(k);
    switch G.kind(t)
        case 1
            if k > 1
                FV(:, :, k) = FV(:, :, k - 1);
            else
                FV(:, :, k) = prev;
            end
        case {2, 3, 4}
            r = G.rows(1, t):G.rows(2, t);
            FV(r, :, k) = uint8(G.gain(t) * double(FV(r, :, k)) + G.offset(t));
    end
end
prev = FV(:, :, end);
end
