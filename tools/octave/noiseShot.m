function [noisyImage,theNoise] = noiseShot(sensor)
% Octave shim for ISETCam noiseShot.
%
% Upstream noiseShot calls sensorGet(sensor,'electrons'), which asserts
% when tiny negative voltages appear during some Octave-only tutorial
% paths. For baseline export we reproduce the same shot-noise math while
% deriving electrons directly from the current volts image.

volts = double(sensorGet(sensor, 'volts'));
ag = sensorGet(sensor, 'analog gain');
ao = sensorGet(sensor, 'analog offset');
conversionGain = sensorGet(sensor, 'pixel conversion gain');

electronImage = (volts * ag - ao) ./ conversionGain;
electronImage = max(electronImage, 0);

electronNoise = sqrt(electronImage) .* randn(size(electronImage));

poissonCriterion = 25;
if ~isempty(find(electronImage < poissonCriterion, 1))
    v = (electronImage < poissonCriterion);
    poissonImage = poissrnd(electronImage .* v);
    electronNoise(v) = poissonImage(v) - electronImage(v);
end

noisyImage = conversionGain * round(electronImage + electronNoise);
theNoise = conversionGain * electronNoise;

end
