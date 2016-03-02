# Copyright © 2013-2014, 2016 Martin Ueding <dev@martin-ueding.de>
# Licensed under The MIT License

.PRECIOUS: %.tex %.pdf build/page/%.pdf

SHELL = /bin/bash

tail = tail -n 15

build = _build

on := $(shell tput smso)
off := $(shell tput rmso)

tex := "$(build)/physics601-$(number)-Ueding_Lemmer.tex"
out := "$(build)/physics601-$(number)-Ueding_Lemmer.pdf"

figures_tex := $(wildcard Figures/*.tex)
figures_pdf := $(figures_tex:Figures/%.tex=$(build)/%.pdf)

all: $(out)

crunch: $(build)/template.js

$(build):
	@echo "$(on)Creating build directory$(off)"
	mkdir -p $(build)
	mkdir -p $(build)/page

$(out): $(tex) $(figures_pdf)
	@echo "$(on)Compiling main document$(off)"
	cd $$(dirname $@) \
	    && latexmk -pdflatex='pdflatex -halt-on-error $$O $$S' -pdf $$(basename $<) \
	    2>&1 | $(tail)

$(tex): Template.tex $(build)/template.js
	@echo "$(on)Inserting values into template$(off)"
	../insert.py $^ $@

$(build)/template.js: crunch.py | $(build)
	@echo "$(on)Crunching the numbers$(off)"
	env PYTHONPATH=$$PYTHONPATH:.. ./$<

$(build)/page/%.tex: Figures/%.tex | $(build)
	@echo "$(on)Wrapping figure $< $(off)"
	../tikzpicture_wrap.py $< $@

$(build)/%.pdf: $(build)/page/%.pdf | $(build)
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
