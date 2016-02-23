# Copyright © 2013-2014, 2016 Martin Ueding <dev@martin-ueding.de>
# Licensed under The MIT License

SHELL = /bin/bash

tex = "_build/physics601-$(number)-Ueding_Lemmer.tex"
out = "_build/physics601-$(number)-Ueding_Lemmer.pdf"

all: $(out)

$(out): $(tex)
	cd _build && latexmk -pdflatex='pdflatex -halt-on-error $$O $$S' -pdf $$(basename $<)

_build/template.js: crunch
	mkdir -p _build
	./$<

$(tex): Template.tex _build/template.js
	../insert $^ $@

.PHONY: clean
clean:
	$(RM) *.class *.jar
	$(RM) *.o *.out
	$(RM) *.pyc *.pyo
	$(RM) *.orig
	$(RM) -r _build
