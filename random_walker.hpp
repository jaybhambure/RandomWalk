#include <random>
#include <iostream>
#include <petscvec.h>

class RandomWalker {
private:
    std::random_device rd;
    std::mt19937 generator;
    std::uniform_int_distribution<> distribution;

public:
    RandomWalker() : generator(rd()), distribution(0, 1) {}

    PetscInt getStep() {
        return distribution(generator) == 0 ? -1 : 1; // Generate -1 or 1
    }

    PetscErrorCode take_steps(Vec &vec, PetscInt num_steps) {
        for (PetscInt i = 0; i < num_steps; ++i) {
            PetscInt step = getStep();
            VecSetValue(vec, i, step, INSERT_VALUES);
        }
        VecAssemblyBegin(vec);
        VecAssemblyEnd(vec);
        return 0;
    }
};