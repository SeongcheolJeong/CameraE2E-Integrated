function value = datetime(varargin)
% Minimal Octave shim for MATLAB datetime used by upstream ieSaveSIDataFile.

if nargin == 0
    value = now();
    return;
end

if nargin == 1 && ischar(varargin{1}) && strcmpi(varargin{1}, 'now')
    value = now();
    return;
end

error('datetime shim only supports datetime() and datetime(''now'').');

end
