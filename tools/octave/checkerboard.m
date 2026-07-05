function board = checkerboard(n, p, q)
% Minimal checkerboard shim for Octave parity runs.
%
% Supports the numeric subset used by ISETCam sceneCreate():
%   checkerboard(n)
%   checkerboard(n, p)
%   checkerboard(n, p, q)

if nargin < 1 || isempty(n)
    n = 10;
end
if nargin < 2 || isempty(p)
    p = 4;
end
if nargin < 3 || isempty(q)
    q = p;
end

tile = [ones(n), zeros(n); zeros(n), ones(n)];
board = repmat(tile, p, q);
end
