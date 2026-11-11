function sensor = camera_optics_frame(binned, t, op)
%% The light a recorded frame carries -> what the sensor reads, before noise.
%
%   light  = binned .* V .* D .* gain(t)
%   sensor = light .* (1 + prnu) + offset
%
% binned is the moving scene averaged over each pixel (apply_camera_motion);
% op comes from make_camera_optics, t is the frame within the session. The
% sensor noise and the rounding follow in apply_camera_motion.
%
% Author: Pablo Vergara

light = binned .* op.VD;
if ~isempty(op.gain_trend)
    g = op.gain_trend(t) * (1 + op.gain_global(t));
    if ~isempty(op.gain_maps) && size(op.gain_maps, 3) > 0
        m = reshape(reshape(op.gain_maps, [], size(op.gain_maps, 3)) * op.gain_coef(:, t), ...
            size(binned));
        light = light .* (g * (1 + m));
    else
        light = light * g;
    end
end
sensor = light .* op.P + op.O;
end
