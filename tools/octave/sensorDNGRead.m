function [sensor, info] = sensorDNGRead(fname, varargin)
% Octave-compatible shim for upstream sensorDNGRead when DNG delegates are unavailable.

varargin = ieParamFormat(varargin);
p = inputParser;
p.addRequired('fname', @(x) ischar(x) || isstring(x));
p.addParameter('fullinfo', true, @islogical);
p.addParameter('crop', [], @(x) isnumeric(x));
p.parse(fname, varargin{:});

fname = char(p.Results.fname);
fullInfo = p.Results.fullinfo;
crop = p.Results.crop;

decoded = decodeDNGForOctave(fname);
img = decoded.img;
ieInfo = struct();
ieInfo.isoSpeed = decoded.isoSpeed;
ieInfo.exposureTime = decoded.exposureTime;
ieInfo.blackLevel = decoded.blackLevel;
ieInfo.orientation = decoded.orientation;

if fullInfo
    info = struct();
    info.FileName = fname;
    info.Orientation = ieInfo.orientation;
    info.BlackLevel = ieInfo.blackLevel;
    info.DigitalCamera = struct( ...
        'ISOSpeedRatings', ieInfo.isoSpeed, ...
        'ExposureTime', ieInfo.exposureTime);
else
    info = ieInfo;
end

blackLevel = ceil(double(ieInfo.blackLevel(1)));
exposureTime = double(ieInfo.exposureTime);
img = ieClip(double(img), blackLevel, []);
isoSpeed = double(ieInfo.isoSpeed);

load(fullfile(isetRootPath, 'data', 'sensor', 'sony', 'imx363.mat'), 'sensor');
sensor = sensorSet(sensor, 'size', size(img));
sensor = sensorSet(sensor, 'exp time', exposureTime);
sensor = sensorSet(sensor, 'black level', blackLevel);
sensor = sensorSet(sensor, 'name', fname);
sensor = sensorSet(sensor, 'digital values', img);

switch ieInfo.orientation
    case 1
        sensor = sensorSet(sensor, 'pattern', [1 2; 2 3]);
    case 3
        sensor = sensorSet(sensor, 'pattern', [3 2; 2 1]);
    case 6
        sensor = sensorSet(sensor, 'pattern', [2 1; 3 2]);
    case 8
        sensor = sensorSet(sensor, 'pattern', [2 3; 1 2]);
    otherwise
        error('Unknown Orientation value');
end

if ~isempty(crop)
    if length(crop) == 4
        crop = round([crop(2), crop(1), crop(4), crop(3)]);
    elseif length(crop) == 1 && crop > 0 && crop < 1
        sz = sensorGet(sensor, 'size');
        middlePosition = sz / 2;
        rowcol = crop * sz;
        row = middlePosition(1) - rowcol(1) / 2;
        col = middlePosition(2) - rowcol(2) / 2;
        height = rowcol(1);
        width = rowcol(2);
        crop = round([row, col, height, width]);
    else
        error('Bad crop value');
    end
    sensor = sensorCrop(sensor, crop);
end

end


function decoded = decodeDNGForOctave(fname)
toolDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(toolDir));
srcDir = fullfile(repoRoot, 'src');
scriptPath = fullfile(repoRoot, 'tools', 'decode_dng_for_octave.py');

pythonExe = getenv('PYISETCAM_PYTHON');
if isempty(pythonExe)
    pythonExe = fullfile(getenv('HOME'), 'miniforge3', 'envs', 'isetcam-py', 'bin', 'python');
end
if exist(pythonExe, 'file') ~= 2
    pythonExe = 'python3';
end

matPath = [tempname(), '.mat'];
command = sprintf( ...
    'PYTHONPATH="%s" "%s" "%s" "%s" "%s"', ...
    srcDir, pythonExe, scriptPath, fname, matPath);
status = system(command);
if status ~= 0
    error('Failed to decode DNG via Python helper');
end

cleanup = onCleanup(@() deleteIfExists(matPath));
decoded = load(matPath);
end


function deleteIfExists(path)
if exist(path, 'file')
    delete(path);
end
end
