function fullFileName = ieSaveColorFilter(inData, fullFileName)
% Minimal Octave-compatible ieSaveColorFilter shim for parity export.

if nargin < 2 || isempty(fullFileName)
    fullFileName = fullfile(pwd, 'colorFilter.mat');
end

if isfield(inData, 'type') && (strcmp(sensorGet(inData, 'type'), 'ISA') || strcmp(sensorGet(inData, 'type'), 'sensor'))
    wavelength = sensorGet(inData, 'wavelength');
    data = sensorGet(inData, 'colorfilters');
    filterNames = sensorGet(inData, 'filterNames');
    comment = 'No comment';
    save(fullFileName, 'wavelength', 'data', 'comment', 'filterNames');
    return;
end

if ~(isfield(inData, 'data') && isfield(inData, 'wavelength') && isfield(inData, 'filterNames'))
    error('Input data missing fields.  No file written.');
end

wavelength = inData.wavelength;
data = inData.data;
filterNames = inData.filterNames;
if isfield(inData, 'comment')
    comment = inData.comment;
else
    comment = 'No comment';
end

save(fullFileName, 'wavelength', 'data', 'comment', 'filterNames');

fields = fieldnames(inData);
for ii = 1:length(fields)
    name = fields{ii};
    if strcmp(name, 'wavelength') || strcmp(name, 'data') || strcmp(name, 'comment') || strcmp(name, 'filterNames') || strcmp(name, 'units')
        continue;
    end
    value = inData.(name); %#ok<NASGU>
    save(fullFileName, '-append', name);
end

end
