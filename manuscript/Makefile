# Makefile for the colorpath manuscript.
#
#   make            build main.pdf (pdflatex -> bibtex -> pdflatex x2)
#   make figures    regenerate figures/ from a parquet (set DATA=...)
#   make clean      remove LaTeX build by-products (keep the PDF)
#   make distclean  also remove main.pdf
#
# Requires a LaTeX distribution (pdflatex, bibtex) on PATH.

MAIN    := main
LATEX   := pdflatex
LFLAGS  := -interaction=nonstopmode -halt-on-error
BIBTEX  := bibtex

# Path to the source parquet for `make figures` (override on the command line).
DATA    ?= /path/to/01_pd_51_raw_by_metabolite_5ppm.parquet

AUX := $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc

.PHONY: all figures clean distclean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex ref.bib figures/components_overview.png
	$(LATEX) $(LFLAGS) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(LFLAGS) $(MAIN)
	$(LATEX) $(LFLAGS) $(MAIN)

figures:
	python make_figures.py $(DATA)

clean:
	rm -f $(AUX)

distclean: clean
	rm -f $(MAIN).pdf
