# Copyright © 2013-2014, 2016 Martin Ueding <dev@martin-ueding.de>
# Licensed under The MIT License

# Do not throw away intermediate results.
.PRECIOUS: %.tex %.pdf build/page/%.pdf build/page/%.tex

# Set the shell to Bash, does not make any difference, probably.
SHELL = /bin/bash

op :=

# Print at most $(lines) from the LaTeX runs, see usage below.
lines = 20
tail = tail -n $(lines)

# Build directory.
build = _build

# Emphasis format strings for turning emphasis on and off.
on := $(shell tput smso)
off := $(shell tput rmso)
green := $(shell tput setaf 2)
red := $(shell tput setaf 1)
reset := $(shell tput sgr0)

# Main document.
tex := "$(build)/physics601-$(number)-Ueding_Lemmer.tex"
out := "$(build)/physics601-$(number)-Ueding_Lemmer.pdf"

# LaTeX and PDF filenames for the figures, plots and Feynman graphs.
figures_tex := $(wildcard Figures/*.tex)
figures_pdf := $(figures_tex:Figures/%.tex=$(build)/%.pdf)
feynman_tex := $(wildcard Feynman/*.tex)
feynman_pdf := $(feynman_tex:Feynman/%.tex=$(build)/%.pdf)
plots_tex := $(wildcard Plots/*.tex)
plots_pdf := $(plots_tex:Plots/%.tex=$(build)/%.pdf)
plots_page_pdf := $(plots_tex:Plots/%.tex=$(build)/page/%.pdf)
plots_page_tex := $(plots_tex:Plots/%.tex=$(build)/page/%.tex)

postscript_ps := $(wildcard Postscript/*.ps)
postscript_pdf := $(postscript_ps:Postscript/%.ps=$(build)/%.pdf)

to_crop_in = $(wildcard $(build)/to_crop/*.pdf)
to_crop_out = $(to_crop_in:$(build)/to_crop/%=$(build)/%)

# The default target is the main PDF document of the report.
all: show-distribution $(out)

test:
	env PYTHONPATH=$$PYTHONPATH:.. python3 -m doctest crunch.py

test-vars:
	@echo "$(on)to_crop_in$(off)"
	@echo $(to_crop_in)
	@echo "$(on)to_crop_out$(off)"
	@echo $(to_crop_out)


$(out): $(to_crop_out)

open:
	xdg-open $(out)

# Running `make crunch` will only run `crunch.py`, nothing more. This is handy
# if has output from the Python program but does not want to typeset the
# report.
crunch: $(build)/template.js

# In case this is running on Fedora, it will be Martin's computer. Then it has
# a recent version of LuaLaTeX and TikZ and can compile the Feynman diagrams
# with `tikz-feynman`. If it is not Fedora, it is Lino's computer where the
# versions are not recent enough.
ifneq "$(wildcard /etc/fedora-release)" ""
    $(out): $(feynman_pdf)

show-distribution:
	@echo "This is running on Fedora. $(green)Typesetting Feynman diagrams.$(reset)"
else
show-distribution:
	@echo "This is running on something other than Fedora. $(red)Not typesetting Feynman diagrams.$(reset)"
endif

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

$(build)/to_crop:
	@echo "$(on)Creating directory for PDF to crop$(off)"
	mkdir -p $(build)/to_crop

# The main document depends on the main LaTeX file as well as all the rendered
# TikZ graphics.
$(out): $(tex) $(figures_pdf) $(plots_pdf) $(postscript_pdf)
	@echo "$(on)Compiling main document$(off)"
	cd $$(dirname $@) \
	    && latexmk -pdflatex='pdflatex -halt-on-error $$O $$S' -pdf $$(basename $<) \
	    2>&1 | $(tail)

# The main LaTeX file is generated using the template engine.
$(tex): Template.tex $(build)/template.js
	@echo "$(on)Inserting values into template$(off)"
	../insert.py $^ $@

# Each plot depends on some data files which are generated by `crunch.py`. This
# means that the analysis should have been run, therefore the first target.
# Also the files with the XY-data could have changed. Here a wildcard of all
# CSV and TSV files is used. The LaTeX compilation is done with `latexmk` and
# therefore that will be intelligent to only run the LaTeX engine if one of the
# included files is changed.
$(plots_page_pdf): $(build)/template.js $(wildcard $(build)/xy/*.?sv)

$(build)/template.js: crunch.py $(wildcard Data/*.*) | $(build)/xy $(build)/to_crop
	@echo "$(on)Crunching the numbers$(off)"
	env PYTHONPATH=$$PYTHONPATH:.. ./$< $(op)

# TikZ figures need to be wrapped into a full document.
$(build)/page/%.tex: Figures/%.tex | $(build)/page
	@echo "$(on)Wrapping figure $<$(off)"
	../tikzpicture_wrap.py $< $@

# TikZ plots need to be wrapped into a full document.
$(build)/page/%.tex: Plots/%.tex | $(build)/page
	@echo "$(on)Wrapping plot $<$(off)"
	../tikzpicture_wrap.py $< $@

# TikZ Feynman diagrams need to be wrapped into a full document with the
# `tikz-feynman` package.
$(build)/page-lualatex/%.tex: Feynman/%.tex | $(build)/page-lualatex
	@echo "$(on)Wrapping plot $<$(off)"
	../tikzpicture_wrap.py --packages tikz-feynman -- $< $@

# Feynman diagrams must be compiled with LuaLaTeX.
$(build)/page-lualatex/%.pdf: $(build)/page-lualatex/%.tex
	@echo "$(on)Typesetting Feynman diagram $<$(off)"
	cd $$(dirname $@) \
	    && latexmk -pdflatex='lualatex -halt-on-error $$O $$S' -pdf $$(basename $<) \
	     2>&1 |$(tail)

# Figures and plots are typeset on an A4 page and therefore need to be cropped.
$(build)/%.pdf: $(build)/page/%.pdf | $(build)/page
	@echo "$(on)Cropping figure $<$(off)"
	pdfcrop $< $@

# Feynman diagrams are typeset on an A4 page and therefore need to be cropped.
$(build)/%.pdf: $(build)/page-lualatex/%.pdf | $(build)/page
	@echo "$(on)Cropping figure $<$(off)"
	pdfcrop $< $@

$(build)/%.pdf: $(build)/to_crop/%.pdf
	@echo "$(on)Cropping matplotlib figure $<$(off)"
	pdfcrop $< $@


# Figures and plots (but not Feynman diagrams) can be compiled using pdflatex.
%.pdf: %.tex
	@echo "$(on)Typesetting figure $<$(off)"
	cd $$(dirname $@) \
	    && latexmk -pdflatex='pdflatex -halt-on-error $$O $$S' -pdf $$(basename $<) \
	     2>&1 |$(tail)

$(build)/%.pdf: Postscript/%.ps
	@echo "$(on)Converting PS file $<$(off)"
	ps2pdf $< /tmp/ps2pdf_$$(basename $@)
	pdfcrop /tmp/ps2pdf_$$(basename $@) $@
	rm -f /tmp/ps2pdf_$$(basename $@)

.PHONY: clean
clean:
	$(RM) *.o *.out
	$(RM) *.pyc *.pyo
	$(RM) *.orig
	$(RM) -r $(build)
