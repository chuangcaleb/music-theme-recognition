from tabulate import tabulate


# custom_headers = ['metric', 'value']

# def latextab_per_label_per_cls(dict):

#     rows = [[key, value] for key, value in dict.items()]
#     table = tabulate(rows, headers=custom_headers, tablefmt='latex')

#     return table


LATEX_TABLE_BEGIN = '\\begin{table}[ht]\n'
LATEX_TABLE_END = '\n\\end{table}'


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
