from statistics import mean, stdev
from matplotlib.font_manager import json_dump
from eval_utils.results_stats_utils import calc_stats
from eval_utils import load_data as data
from eval_utils.export_results import json_dump, results_table_dump

avg_results_dict = calc_stats(data.all_results_dict, mean)
json_dump(avg_results_dict, 'results_avg', 'results/')
results_table_dump(avg_results_dict, 'results_avg', 'Average')

std_results_dict = calc_stats(data.all_results_dict, stdev)
json_dump(std_results_dict, 'results_std', 'results/')
results_table_dump(std_results_dict, 'results_std', 'Standard Deviation')
