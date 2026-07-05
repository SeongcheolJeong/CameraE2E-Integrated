function scene = sceneCombine(scene1, scene2, varargin)
% Minimal sceneCombine shim for Octave parity runs.

direction = 'horizontal';
if nargin > 2
    if numel(varargin) ~= 2 || ~strcmpi(varargin{1}, 'direction')
        error('sceneCombine expects ''direction'', value arguments');
    end
    direction = lower(varargin{2});
end

if ~isequal(sceneGet(scene1, 'wave'), sceneGet(scene2, 'wave'))
    error('sceneCombine requires matching wavelength samples');
end

switch direction
    case 'horizontal'
        if sceneGet(scene1, 'rows') ~= sceneGet(scene2, 'rows')
            error('Horizontal sceneCombine requires matching row counts');
        end
        photons = [sceneGet(scene1, 'photons'), sceneGet(scene2, 'photons')];
        scene = sceneSet(scene1, 'photons', photons);
        scene = sceneSet(scene, 'hfov', sceneGet(scene1, 'fov') + sceneGet(scene2, 'fov'));

    case 'vertical'
        if sceneGet(scene1, 'cols') ~= sceneGet(scene2, 'cols')
            error('Vertical sceneCombine requires matching column counts');
        end
        photons = [sceneGet(scene1, 'photons'); sceneGet(scene2, 'photons')];
        scene = sceneSet(scene1, 'photons', photons);

    case 'both'
        scene = sceneCombine(scene1, scene2, 'direction', 'horizontal');
        scene = sceneCombine(scene, scene, 'direction', 'vertical');

    case 'centered'
        sceneMid = sceneCombine(scene1, scene2, 'direction', 'horizontal');
        sceneMid = sceneCombine(scene2, sceneMid, 'direction', 'horizontal');
        sceneEdge = sceneCombine(scene2, scene2, 'direction', 'horizontal');
        sceneEdge = sceneCombine(sceneEdge, scene2, 'direction', 'horizontal');
        scene = sceneCombine(sceneEdge, sceneMid, 'direction', 'vertical');
        scene = sceneCombine(scene, sceneEdge, 'direction', 'vertical');

    otherwise
        error('Unsupported sceneCombine direction');
end
end
