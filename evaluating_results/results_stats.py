from statistics import mean, stdev

from eval_utils import load_results as data
from eval_utils.results_stats_utils import calc_and_dump


avg_results_dict = calc_and_dump(data.all_results_dict, mean, 'results_avg',
                                 'Average')
std_results_dict = calc_and_dump(data.all_results_dict, stdev, 'results_std',
                                 'Standard Deviation')
