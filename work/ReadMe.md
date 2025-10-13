# 1d random walker
```
files - 1drw.cpp, initialization.hpp, single_walker.hpp, 1drw_plot.py

mpicxx -o 1drw 1drw.cpp `pkg-config --cflags petsc` `pkg-config --libs petsc`
mpiexec -n 1 ./1drw
python3 1drw_plot.py
```


# Rough file to work with
```
mpicxx -o delete delete.cpp `pkg-config --cflags petsc` `pkg-config --libs petsc`
mpiexec -n 1 ./delete
```