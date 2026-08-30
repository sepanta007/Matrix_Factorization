# Local build of the LaTeX report.
# Usage:
#   make        -> build report.pdf
#   make quick  -> single pdflatex pass (fast iteration, no biblio/refs)
#   make clean  -> remove auxiliary files
#   make purge  -> clean + remove the PDF

LATEX  = pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX = bibtex

.PHONY: all quick clean purge

all:
	$(LATEX) report.tex
	-$(BIBTEX) report
	$(LATEX) report.tex
	$(LATEX) report.tex
	@echo "==> report.pdf generated."

quick:
	$(LATEX) report.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot \
	      *.fls *.fdb_latexmk *.synctex.gz chapters/*.aux

purge: clean
	rm -f report.pdf
