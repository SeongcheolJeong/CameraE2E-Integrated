function fName = ieSaveSIDataFile(psf, wave, umPerSamp, fName)
% Octave-compatible shim for upstream ieSaveSIDataFile.

if ~exist('psf', 'var') || isempty(psf), error('psf volume required'); end
if ~exist('wave', 'var') || isempty(wave), error('wavelength samples required (nm)'); end
if ~exist('umPerSamp', 'var') || isempty(umPerSamp), error('Microns per sample(2-vector) required'); end
if ~exist('fName', 'var') || isempty(fName), error('output file name required in headless Octave shim'); end

notes.timeStamp = now();
save('-mat7-binary', fName, 'psf', 'wave', 'umPerSamp', 'notes');

end
