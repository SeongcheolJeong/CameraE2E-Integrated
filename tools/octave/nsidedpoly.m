function pgon = nsidedpoly(nSides, varargin)
% Minimal nsidedpoly shim for curated Codex parity cases.

center = [0, 0];
radius = 1;

for ii = 1:2:numel(varargin)
    key = lower(strrep(varargin{ii}, ' ', ''));
    value = varargin{ii + 1};
    switch key
        case 'center'
            center = value;
        case 'radius'
            radius = value;
        otherwise
            error('nsidedpoly:unsupportedOption', 'Unsupported option %s', varargin{ii});
    end
end

theta = linspace(0, 2*pi, nSides + 1);
theta(end) = [];
x = center(1) + radius * cos(theta(:));
y = center(2) + radius * sin(theta(:));
pgon.Vertices = [x, y];
end
