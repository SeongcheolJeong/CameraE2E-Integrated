function tf = contains(str, pattern, varargin)
% Minimal Octave shim for MATLAB contains().

if nargin < 2
    error('contains requires str and pattern');
end

ignoreCase = false;
idx = 1;
while idx <= numel(varargin)
    key = varargin{idx};
    if idx == numel(varargin)
        error('contains: invalid key/value arguments');
    end
    value = varargin{idx + 1};
    if ischar(key) && strcmpi(key, 'IgnoreCase')
        ignoreCase = logical(value);
    end
    idx = idx + 2;
end

if iscell(str)
    tf = cellfun(@(s) local_contains_char(s, pattern, ignoreCase), str);
    return;
end

tf = local_contains_char(str, pattern, ignoreCase);


function out = local_contains_char(strValue, patternValue, ignoreCaseValue)
if iscell(patternValue)
    out = any(cellfun(@(p) local_contains_char(strValue, p, ignoreCaseValue), patternValue));
    return;
end

if ignoreCaseValue
    strValue = lower(char(strValue));
    patternValue = lower(char(patternValue));
else
    strValue = char(strValue);
    patternValue = char(patternValue);
end

out = ~isempty(strfind(strValue, patternValue));
