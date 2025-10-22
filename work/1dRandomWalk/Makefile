# Set PETSc directories directly
PETSC_DIR = /home/jay/petsc
PETSC_ARCH = arch-linux-c-debug

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

randomWalk1d: randomWalk1d.o
	-${CLINKER} -o randomWalk1d randomWalk1d.o ${PETSC_LIB}
	${RM} randomWalk1d.o

run: randomWalk1d
	@echo "Running random walk simulation..."
	@./randomWalk1d

plot:
	@echo "Generating plots from HDF5 data..."
	@python3 randomWalk_plot.py

# Compile LaTeX documentation (keep only PDF, remove auxiliary files)
tex:
	@echo "Compiling LaTeX documentation..."
	@pdflatex randomwalk_notes.tex > /dev/null 2>&1
	@pdflatex randomwalk_notes.tex > /dev/null 2>&1
	@rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz
	@echo "PDF documentation generated: randomwalk_notes.pdf"

# Remove all generated files (avoids conflict with PETSc clean)
remove:
	@echo "Removing all generated files..."
	@rm -f randomWalk1d *.o *.h5 *.png
	@rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz
	@echo "All files removed"

.PHONY: run plot tex remove