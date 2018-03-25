from datetime import datetime


def gen_result_dir(profile=0):
	if profile == 0:
		now = datetime.now()
		name = "result_{}_{}_{}_{}_{}_{}/"\
		.format(
			now.year,
			now.month,
			now.day,
			now.hour,
			now.minute,
			now.second
			)
		return name
