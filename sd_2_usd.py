'''
OBJECTIVES:
    Define a mapping from Stanford dependencies to Universal dependencies.
NOTE:
    We found that in many cases 'pobj' are actually salient in core clause structures. Thus, even though in the paper
    "Universal Stanford Dependencies: A cross-linguistic typology" it is suggested to convert 'pobj' to 'nmod', we do
    not do this in our project.
'''

g_d_sd_to_usd = {'acomp': 'xcomp',
                 'attr': 'xcomp',
                 'purpcl': 'advcl',
                 'abbrev': 'appos',
                 'num': 'nummod',
                 'quantmod': 'advmod',
                 'complm': 'mark',
                 'rel': 'dep',
                 'prep': 'case',
                 #'pobj': 'nmod',
                 'possessive': 'case',
                 'nn': 'compound',
                 'number': 'compound',
                 'npadvmod': 'nmod',
                 'tmod': 'nmod',
                 'predet': 'det',
                 'preconj': 'cc',
                 'prt': 'compound',
                 'poss': 'case'}


def sd_to_usd(dep):
    if dep in g_d_sd_to_usd:
        return g_d_sd_to_usd[dep]
    else:
        return dep