# Makefile for the GAIT manuscript.
#
#   make            build main.pdf and supplement.pdf
#   make figures    regenerate figures/ from a parquet (set DATA=...)
#   make clean      remove LaTeX build by-products (keep the PDFs)
#   make distclean  also remove the PDFs
#
# Requires a LaTeX distribution (pdflatex, bibtex) on PATH.

MAIN    := main
SUPP    := supplement
LATEX   := pdflatex
LFLAGS  := -interaction=nonstopmode -halt-on-error
BIBTEX  := bibtex

# Path to the source parquet for `make figures` (override on the command line).
DATA    ?= /path/to/01_pd_51_raw_by_metabolite_5ppm.parquet

AUX := $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc \
       $(SUPP).aux $(SUPP).log $(SUPP).out

.PHONY: all figures clean distclean

all: $(MAIN).pdf $(SUPP).pdf

$(MAIN).pdf: $(MAIN).tex ref.bib figures/components_overview.png
	$(LATEX) $(LFLAGS) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(LFLAGS) $(MAIN)
	$(LATEX) $(LFLAGS) $(MAIN)

$(SUPP).pdf: $(SUPP).tex
	$(LATEX) $(LFLAGS) $(SUPP)
	$(LATEX) $(LFLAGS) $(SUPP)

figures:
	python make_figures.py $(DATA)

clean:
	rm -f $(AUX)

distclean: clean
	rm -f $(MAIN).pdf $(SUPP).pdf
