function [value, img] = airyDisk(this_wave, f_number, varargin)
% Headless Octave shim for upstream airyDisk.m.
%
% The upstream helper routes image output through oiPlot/ieFigure, which is
% not usable in this parity harness. This shim preserves the numeric API used
% by the curated export cases and script-driven tests.

units = 'um';
diameter = false;
pupil_diameter_m = [];

idx = 1;
while idx <= numel(varargin)
    key = ieParamFormat(varargin{idx});
    if idx == numel(varargin)
        error('airyDisk: invalid key/value arguments');
    end
    val = varargin{idx + 1};
    switch key
        case 'units'
            units = val;
        case 'diameter'
            diameter = logical(val);
        case {'pupildiameter', 'pupilsize'}
            pupil_diameter_m = double(val);
        otherwise
            error('airyDisk: unsupported option %s', key);
    end
    idx = idx + 2;
end

wave_m = double(this_wave);
if wave_m > 200
    wave_m = wave_m * 1e-9;
end

normalized_units = ieParamFormat(units);
if isempty(f_number)
    if isempty(pupil_diameter_m)
        error('airyDisk: pupil diameter is required for angular outputs');
    end
    radius = asin(1.22 * wave_m / max(double(pupil_diameter_m), 1e-12));
    switch normalized_units
        case {'rad', 'radian', 'radians'}
            value = radius;
        case {'deg', 'degree', 'degrees'}
            value = radius * 180 / pi;
        otherwise
            error('airyDisk: unsupported angular unit %s', units);
    end
    if diameter
        value = 2 * value;
    end
    if nargout > 1
        img = [];
    end
    return;
end

radius_m = 1.22 * double(f_number) * wave_m;
switch normalized_units
    case {'m', 'meter', 'meters'}
        value = radius_m;
    case {'mm', 'millimeter', 'millimeters'}
        value = radius_m * 1e3;
    case {'um', 'micron', 'microns'}
        value = radius_m * 1e6;
    otherwise
        error('airyDisk: unsupported spatial unit %s', units);
end
if diameter
    value = 2 * value;
end

if nargout > 1
    focal_length_mm = 22.0;
    wvf = wvfCreate();
    wvf = wvfSet(wvf, 'wave', double(this_wave));
    wvf = wvfSet(wvf, 'measured wavelength', double(this_wave));
    wvf = wvfSet(wvf, 'focal length', focal_length_mm, 'mm');
    wvf = wvfSet(wvf, 'calc pupil diameter', focal_length_mm / max(double(f_number), 1e-12), 'mm');
    wvf = wvfCompute(wvf);
    psf = wvfGet(wvf, 'psf', double(this_wave));
    x = wvfGet(wvf, 'psf spatial samples', 'um', double(this_wave));
    img = struct('data', psf, 'x', x(:)', 'y', x(:)');
end
