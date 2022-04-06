import json
import pickle
from tabulate import tabulate


# * Dump -----------------------------------------------------------------------

def pickle_dump(dict, filename):
    pickle.dump(dict, open(OUTPUT_PATH + filename + ".pickle", "wb"))


def json_dump(dict, filename):
    json.dump(dict, open(OUTPUT_PATH + filename + ".json", "w"))


def best_results_table_dump(best_results_dict):

    output_latex_tables = {}
    output_md_tables = {}

    for current_label in best_results_dict:

        output_latex_tables[current_label], output_md_tables[current_label] = latextab_per_label(
            best_results_dict[current_label], current_label)

    tables_dump(output_latex_tables, 'latex_tables')
    tables_dump(output_md_tables, 'md_tables')


def tables_dump(output_tables, filename):

    with open(OUTPUT_PATH + filename + ".txt", "w") as f:

        for tableId in output_tables:

            f.write(tableId + '\n\n' + output_tables[tableId] + '\n\n\n')

        f.close()


# * LATEX ----------------------------------------------------------------------

LATEX_TABLE_BEGIN = '\\begin{table}[ht]\n'
LATEX_TABLE_END = '\n\\end{table}'
OUTPUT_PATH = 'data/output/'


def latextab_per_label(dict, label):

    rows = [[key] + list(dict[key].values()) for key, value in dict.items()]
    headers = list(dict[list(dict)[0]].keys())

    latex_table = tabulate(rows, headers=headers, tablefmt='latex')
    markdown_table_output = tabulate(rows, headers=headers, tablefmt='github')

    print(f'\n{label}')
    print(markdown_table_output)

    latex_table_output = build_latex_table(latex_table, label)

    return latex_table_output, markdown_table_output


def build_latex_table(table, label):
    return LATEX_TABLE_BEGIN + table + build_latex_caption(label) + LATEX_TABLE_END


def build_latex_caption(label):
    return f'\n\caption{{\\label{{tab: results-{label}}} Model performances for \'{label}\'.}}'
