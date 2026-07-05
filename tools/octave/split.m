function parts = split(str, delimiter)
if nargin < 2
    delimiter = ' ';
end

if isstring(str)
    str = char(str);
elseif iscell(str) && numel(str) == 1
    str = str{1};
end

parts = strsplit(str, delimiter);
parts = parts(:);
end
