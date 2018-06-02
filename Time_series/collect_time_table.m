% Collect time timetable values from csv.

% Name of variable to be loaded into workspace.
names = {
	'UNRATE',
	'M2NS',
	'M2',
	'CPIAUCSL'
	};

collect = {};

% Load data to workspace.
for i = 1: length(names)
	table = readtable([names{i}, '.csv']);
	tt = table2timetable(table);
	collect{i} = tt;
end

data_tt = synchronize(collect{:});
data_tt = rmmissing(data_tt); % Remove missing variables.

data_tb = timetable2table(data_tt);
data_ar = table2array(data_tb(:, 2:end))'; % Collection of numerical data
% with DATE variable removed.

