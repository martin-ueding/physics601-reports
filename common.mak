# Copyright © 2013-2014, 2016 Martin Ueding <dev@martin-ueding.de>
# Licensed under The MIT License

.PRECIOUS: %.tex %.pdf build/page/%.pdf build/page/%.tex

SHELL = /bin/bash

tail = tail -n 20

build = _build

on := $(shell tput smso)
off := $(shell tput rmso)

tex := "$(build)/physics601-$(number)-Ueding_Lemmer.tex"
out := "$(build)/physics601-$(number)-Ueding_Lemmer.pdf"

figures_tex := $(wildcard Figures/*.tex)
figures_pdf := $(figures_tex:Figures/%.tex=$(build)/%.pdf)
feynman_tex := $(wildcard Feynman/*.tex)
feynman_pdf := $(feynman_tex:Feynman/%.tex=$(build)/%.pdf)

plots_tex := $(wildcard Plots/*.tex)
plots_pdf := $(plots_tex:Plots/%.tex=$(build)/%.pdf)
plots_page_pdf := $(plots_tex:Plots/%.tex=$(build)/page/%.pdf)
plots_page_tex := $(plots_tex:Plots/%.tex=$(build)/page/%.tex)

all: $(out)

crunch: $(build)/template.js

$(build):
	@echo "$(on)Creating build directory$(off)"
	mkdir -p $(build)

$(build)/page:
	@echo "$(on)Creating directory for full pages$(off)"
	mkdir -p $(build)/page

$(build)/page-lualatex:
	@echo "$(on)Creating build directory for full pages (LuaLaTeX)$(off)"
	mkdir -p $(build)/page-lualatex

$(build)/xy:
	@echo "$(on)Creating directory for X-Y data$(off)"
	mkdir -p $(build)/xy

$(out): $(tex) $(figures_pdf) $(plots_pdf) $(feynman_pdf)
	@echo "$(on)Compiling main document$(off)"
	cd $$(dirname $@) \
	    && latexmk -pdflatex='pdflatex -halt-on-error $$O $$S' -pdf $$(basename $<) \
	    2>&1 | $(tail)

$(tex): Template.tex $(build)/template.js
	@echo "$(on)Inserting values into template$(off)"
	../insert.py $^ $@

$(plots_page_pdf): $(build)/template.js $(wildcard $(build)/xy/*.csv)

$(build)/template.js: crunch.py Data/* | $(build)/xy
	@echo "$(on)Crunching the numbers$(off)"
	env PYTHONPATH=$$PYTHONPATH:.. ./$<

$(build)/page/%.tex: Figures/%.tex | $(build)/page
	@echo "$(on)Wrapping figure $< $(off)"
	../tikzpicture_wrap.py $< $@

$(build)/page/%.tex: Plots/%.tex | $(build)/page
	@echo "$(on)Wrapping plot $< $(off)"
	../tikzpicture_wrap.py $< $@

$(build)/page-lualatex/%.tex: Feynman/%.tex | $(build)/page-lualatex
	@echo "$(on)Wrapping plot $< $(off)"
	../tikzpicture_wrap.py --packages tikz-feynman -- $< $@

$(build)/page-lualatex/%.pdf: $(build)/page-lualatex/%.tex
	@echo "$(on)Typesetting Feynman diagram $< $(off)"
	cd $$(dirname $@) \
	    && latexmk -pdflatex='lualatex -halt-on-error $$O $$S' -pdf $$(basename $<) \
	     2>&1 |$(tail)

$(build)/%.pdf: $(build)/page/%.pdf | $(build)/page
	@echo "$(on)Cropping figure $< $(off)"
	pdfcrop $< $@

$(build)/%.pdf: $(build)/page-lualatex/%.pdf | $(build)/page
	@echo "$(on)Cropping figure $< $(off)"
	pdfcrop $< $@


%.pdf: %.tex
	@echo "$(on)Typesetting figure $< $(off)"
	cd $$(dirname $@) \
	    && latexmk -pdflatex='pdflatex -halt-on-error $$O $$S' -pdf $$(basename $<) \
	     2>&1 |$(tail)

.PHONY: clean
clean:
	$(RM) *.class *.jar
	$(RM) *.o *.out
	$(RM) *.pyc *.pyo
	$(RM) *.orig
	$(RM) -r $(build)
