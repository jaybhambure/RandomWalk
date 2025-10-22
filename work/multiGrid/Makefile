# Set PETSc directories directly
PETSC_DIR = /home/jay/petsc
PETSC_ARCH = arch-linux-c-debug

# HDF5 flags
HDF5_LIBS = $(shell pkg-config --libs hdf5)

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# Add HDF5 to compilation flags
CPPFLAGS += $(shell pkg-config --cflags hdf5)

fineGrid: fineGrid.o
	-${CLINKER} -o fineGrid fineGrid.o ${PETSC_LIB} ${HDF5_LIBS}
	${RM} fineGrid.o

run: fineGrid
	@echo "Running random walk simulation..."
	@./fineGrid

plot:
	@echo "Generating plots from HDF5 data..."
	@python3 quickPlot.py

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