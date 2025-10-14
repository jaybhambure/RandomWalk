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

.PHONY: run plot