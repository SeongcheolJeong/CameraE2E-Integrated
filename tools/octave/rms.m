function value = rms(x, dim)
% Minimal Octave shim for MATLAB rms used by upstream ISETCam exports.

if nargin < 2
    dim = [];
end

x = double(x);
if isempty(dim)
    value = sqrt(mean(abs(x).^2));
else
    value = sqrt(mean(abs(x).^2, dim));
end

end
