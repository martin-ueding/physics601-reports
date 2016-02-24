#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright © 2013-2014 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2 (or later)

import argparse
import json

import jinja2

def render_template(template_fn, data_fn, output_fn):
    # Setting up Jinja
    env = jinja2.Environment(
        "%<", ">%",
        "<<", ">>",
        "/*", "*/",
        loader=jinja2.FileSystemLoader(".")
    )
    template = env.get_template(template_fn)

    with open(data_fn) as handle:
        data = json.load(handle)

    # Rendering LaTeX document with values.
    with open(output_fn, "w") as handle:
        handle.write(template.render(**data))

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("template", help="LaTeX template with Jinja 2")
    parser.add_argument("data", help="JSON encoded data file")
    parser.add_argument("output", help="Output LaTeX File")

    options = parser.parse_args()

    render_template(options.template, options.data, options.output)

if __name__ == '__main__':
    main()
