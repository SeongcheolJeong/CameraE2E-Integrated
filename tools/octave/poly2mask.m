function mask = poly2mask(x, y, rows, cols)
% Minimal poly2mask shim for curated Codex parity cases.

[xx, yy] = meshgrid(1:cols, 1:rows);
mask = inpolygon(xx, yy, x(:), y(:));
end
