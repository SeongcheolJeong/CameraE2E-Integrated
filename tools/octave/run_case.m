args = argv();
if numel(args) < 3
    error('Usage: octave-cli tools/octave/run_case.m <case_name> <output_path> <upstream_root>');
end

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);

case_name = args{1};
output_path = args{2};
upstream_root = args{3};

warning('off', 'all');
export_case(case_name, output_path, upstream_root);
