# Copyright © 2013-2014, 2016 Martin Ueding <dev@martin-ueding.de>
# Licensed under The MIT License

number = K225

include ../common.mak

pres_tex := "$(build)/physics601-$(number)-presentation-Ueding_Lemmer.tex"
pres_out := "$(build)/physics601-$(number)-presentation-Ueding_Lemmer.pdf"

all: $(pres_out)

$(pres_out): $(pres_tex) $(figures_pdf) $(plots_pdf) $(postscript_pdf)
	@echo "$(on)Compiling presentation$(off)"
	cd $$(dirname $@) \
	    && latexmk -pdflatex='pdflatex -halt-on-error $$O $$S' -pdf $$(basename $<) \
	    2>&1 | $(tail)

$(pres_tex): Presentation.tex $(build)/template.js
	@echo "$(on)Inserting values into presentation$(off)"
	../insert.py $^ $@

ifneq "$(wildcard /etc/fedora-release)" ""
    $(pres_out): $(feynman_pdf)
endif
