# Copyright © 2013-2014, 2016 Martin Ueding <dev@martin-ueding.de>
# Licensed under The MIT License

.PRECIOUS: %.tex %.pdf build/page/%.pdf

SHELL = /bin/bash

build = _build

tex = "$(build)/physics601-$(number)-Ueding_Lemmer.tex"
out = "$(build)/physics601-$(number)-Ueding_Lemmer.pdf"

figures_tex := $(wildcard Figures/*.tex)
figures_pdf := $(figures_tex:Figures/%.tex=$(build)/%.pdf)

all: $(out)

$(build):
	mkdir -p $(build)
	mkdir -p $(build)/page

$(out): $(tex) $(figures_pdf)
	cd $$(dirname $@) && latexmk -pdflatex='pdflatex -halt-on-error $$O $$S' -pdf $$(basename $<)

$(tex): Template.tex $(build)/template.js
	../insert.py $^ $@

$(build)/template.js: crunch | $(build)
	env PYTHONPATH=$PYTHONPATH:.. ./$<

$(build)/page/%.tex: Figures/%.tex | $(build)
	../tikzpicture_wrap.py $< $@

$(build)/%.pdf: $(build)/page/%.pdf | $(build)
	pdfcrop $< $@

%.pdf: %.tex
	cd $$(dirname $@) && latexmk -pdflatex='pdflatex -halt-on-error $$O $$S' -pdf $$(basename $<)

.PHONY: clean
clean:
	$(RM) *.class *.jar
	$(RM) *.o *.out
	$(RM) *.pyc *.pyo
	$(RM) *.orig
	$(RM) -r $(build)
