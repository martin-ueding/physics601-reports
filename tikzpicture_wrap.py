#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright © 2015-2016 Martin Ueding <dev@martin-ueding.de>

import argparse
import os

import jinja2

template_text = r'''
\documentclass[11pt, << lang >>, DIV=15, BCOR=2cm]{<< documentclass >>}

\usepackage[<< extra_opts >>]{../../../header}

\usepackage{tikz}
\tikzset{
    wave/.style={decorate, decoration=snake}
}
\usepackage{pgfplots}
\pgfplotsset{compat=1.9}

%< for package in packages >%
\usepackage{<< package >>}
%< endfor >%

\pagestyle{empty}

\begin{document}

<< snippet >>

\end{document}
'''

def main():
    options = _parse_args()

    env = jinja2.Environment(
        "%<", ">%",
        "<<", ">>",
        "/*", "*/",
    )
    template = env.from_string(template_text)

    with open(options.infile) as f:
        snippet = f.read()

    if options.packages is None:
        options.packages = []

    is_beamer = os.path.basename(options.infile).startswith('beamer-')
    documentclass = 'beamer' if is_beamer else 'scrartcl'
    extra_opts = 'beamer' if is_beamer else ''

    rendered = template.render(lang=options.lang, snippet=snippet,
                               packages=options.packages,
                               documentclass=documentclass,
                               extra_opts=extra_opts)

    with open(options.outfile, 'w') as f:
        f.write(rendered)

def _parse_args():
    '''
    Parses the command line arguments.

    :return: Namespace with arguments.
    :rtype: Namespace
    '''
    parser = argparse.ArgumentParser(description='Wraps a single environment into a standalone document.')
    parser.add_argument('infile', help='Snippet LaTeX file with tikzpicture')
    parser.add_argument('outfile', help='Standalone LaTeX file')
    parser.add_argument('--lang', default='english', help='Language for babel, default is %(default)s')
    parser.add_argument('--packages', nargs='*', help='Additional packages to load')
    options = parser.parse_args()

    return options

if __name__ == '__main__':
    main()
