from matplotlib.font_manager import json_dump
from eval_utils.results_stats_utils import average_results
from eval_utils import load_data as data
from eval_utils.export_results import json_dump, results_table_dump

avg_results_dict = average_results(data.all_results_dict)
json_dump(avg_results_dict, 'avg_results', 'results/')
results_table_dump(avg_results_dict, 'avg', 'Average')
