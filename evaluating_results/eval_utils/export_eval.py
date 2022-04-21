import json
import os
import pickle

from tabulate import tabulate

from eval_utils import config as cfg


# * Dump -----------------------------------------------------------------------


def pickle_dump(dict, filename):
    pickle.dump(dict, open(cfg.RUN_DIR + filename + ".pickle", "wb"))


def json_dump(dict, filename, subdir=''):
    filepath = cfg.RUN_DIR + subdir + filename + ".json"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    json.dump(dict, open(filepath, "w"))


def dump_results(dict, filename, title):
    json_dump(dict, filename, 'results/')
    results_table_dump(dict, filename, title)


def results_table_dump(results_dict, name, caption):
    """ Main function to dump results in tables as text files """

    def round_dict_values(d, k):
        """ Round all values in dictionary to k decimal places """

        return {key: '{:.03f}'.format(d[key]) for key in d}

    output_latex_tables = {}
    output_md_tables = {}

    print(f'\n\n> \033[93m{caption}\033[0m results')

    for current_label in results_dict:

        current_results = results_dict[current_label]

        # Round results
        rounded_current_results = {
            metric: round_dict_values(current_results[metric], 3)
            for metric in current_results
        }

        # Build latex and md tables
        output_latex_tables[current_label], output_md_tables[current_label] = build_label_table(
            rounded_current_results, current_label, caption)

    # Write tables to files
    tables_txt_dump(output_latex_tables, caption, f'latex/{name}.tex')
    tables_txt_dump(output_md_tables, caption,  f'md/{name}.md')


# * HELPER ---------------------------------------------------------------------


def tables_txt_dump(output_tables, caption, relpath):
    """ Helper function to write tables to text files """

    filepath = cfg.RUN_DIR + 'tables/' + relpath
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w") as f:

        f.write(f'# {cfg.RUN_ID}: {caption} results\n')

        for tableId in output_tables:

            f.write('\n## ' + tableId + '\n\n' + output_tables[tableId] + '\n')

        f.close()


def build_label_table(dict, label, caption):
    """ Helper function to build the md and latex result tables """

    # ------------------------- Helper Functions ----------------------------- #

    LATEX_TABLE_BEGIN = '\\begin{table}[ht]\n'
    LATEX_TABLE_END = '\n\\end{table}'

    def build_latex_table(table, label, caption):
        """ Helper function to build the latex wrappers around the table """

        return LATEX_TABLE_BEGIN + table + build_latex_caption(label, caption) + LATEX_TABLE_END

    def build_latex_caption(label, caption):
        """ Helper function to build the latex caption """

        return f'\n\caption{{\\label{{tab: results-{label}}} {caption} model performances for `{label}\'.}}'

    # ------------------------------ Code ----------------------------------- #

    rows = [
        [key] + list(dict[key].values()) for key in dict.keys()
    ]
    headers = list(
        dict[list(dict)[0]].keys()
    )

    latex_table = tabulate(
        rows,
        headers=headers,
        tablefmt='latex',
        disable_numparse=True
    )
    markdown_table_output = tabulate(
        rows,
        headers=headers,
        tablefmt='github',
        numalign="left",
        disable_numparse=True
    )

    print(f'\n{label}\n')
    print(markdown_table_output)

    latex_table_output = build_latex_table(latex_table, label, caption)

    return latex_table_output, markdown_table_output
