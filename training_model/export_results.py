import json
import pickle
from tabulate import tabulate

# * LATEX

LATEX_TABLE_BEGIN = '\\begin{table}[ht]\n'
LATEX_TABLE_END = '\n\\end{table}'
OUTPUT_PATH = 'data/output/'


def latextab_per_label(dict, label):

    rows = [[key] + list(dict[key].values()) for key, value in dict.items()]
    headers = list(dict[list(dict)[0]].keys())
    table = tabulate(rows, headers=headers, tablefmt='latex')

    print('\n' + tabulate(rows, headers=headers, tablefmt='github'))
    output = build_latex_table(table, label)

    return output


def build_latex_table(table, label):
    return LATEX_TABLE_BEGIN + table + build_latex_caption(label) + LATEX_TABLE_END


def build_latex_caption(label):
    return f'\n\caption{{\\label{{tab: {label}}} Model performances for \'{label}\'.}}'


# * Dump


def models_dump(output_models_dict):
    pickle.dump(
        output_models_dict,
        open(OUTPUT_PATH + "output_models.pickle", "wb")
    )


def results_dump(output_results_dict):
    json.dump(
        output_results_dict,
        open(OUTPUT_PATH + "output_results.json", "w")
    )
