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

    latex_table = tabulate(rows, headers=headers, tablefmt='latex')
    markdown_table_output = tabulate(rows, headers=headers, tablefmt='github')

    print('\n' + markdown_table_output)
    latex_table_output = build_latex_table(latex_table, label)

    return latex_table_output, markdown_table_output


def build_latex_table(table, label):
    return LATEX_TABLE_BEGIN + table + build_latex_caption(label) + LATEX_TABLE_END


def build_latex_caption(label):
    return f'\n\caption{{\\label{{tab: results-{label}}} Model performances for \'{label}\'.}}'


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


def tables_dump(output_latex_tables, output_md_tables):
    with open(OUTPUT_PATH + "latex_tables.txt", "w") as f:

        for tableId in output_latex_tables:

            f.write('\n\n\n' + tableId + '\n\n' + output_latex_tables[tableId])

        f.close()

    with open(OUTPUT_PATH + "md_tables.txt", "w") as f:

        for tableId in output_md_tables:

            f.write('\n\n\n' + tableId + '\n\n' + output_md_tables[tableId])

        f.close()
