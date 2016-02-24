#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright © 2015 Martin Ueding <dev@martin-ueding.de>

import argparse

import jinja2

template_text = r'''
\documentclass[11pt, {{ lang }}]{scrartcl}

\usepackage{../../../header}

\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.10}

\pagestyle{empty}

\begin{document}

{{ snippet }}

\end{document}
'''

def main():
    options = _parse_args()

    template = jinja2.Template(template_text)

    with open(options.infile) as f:
        snippet = f.read()

    rendered = template.render(lang=options.lang, snippet=snippet)

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
    options = parser.parse_args()

    return options

if __name__ == '__main__':
    main()
