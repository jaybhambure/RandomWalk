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
	@python3 plot_results.py

# Compile LaTeX documentation (keep only PDF, remove auxiliary files)
tex:
	@echo "Compiling LaTeX documentation..."
	@pdflatex randomwalk_notes.tex > /dev/null 2>&1
	@pdflatex randomwalk_notes.tex > /dev/null 2>&1
	@rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz
	@echo "✅ PDF documentation generated: randomwalk_notes.pdf"

# Clean up all generated files (extends PETSc clean rule)
cleanall: clean
	@echo "Cleaning up HDF5 data and LaTeX auxiliary files..."
	@rm -f *.h5
	@rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz
	@echo "✅ Complete cleanup finished"

.PHONY: run plot tex cleanall