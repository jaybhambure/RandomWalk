#ifndef INITIALIZATION_HPP
#define INITIALIZATION_HPP

#include <petscvec.h>
#include <random>
#include <cmath>

class InitialPositions {
private:
    std::random_device rd;
    std::mt19937 generator;
    std::uniform_real_distribution<> uniform_dist;
    std::normal_distribution<> gaussian_dist;

public:
    InitialPositions() : generator(rd()), uniform_dist(0.0, 1.0), gaussian_dist(0.0, 1.0) {}

    // Uniform distribution across the grid
    PetscErrorCode uniform(Vec &vec, PetscInt gridSize, PetscInt localStart, PetscInt localEnd) {
        PetscFunctionBegin;
        PetscScalar *array;
        VecGetArray(vec, &array);

        for (PetscInt i = localStart; i < localEnd; ++i) {
            array[i - localStart] = i % gridSize;
        }

        VecRestoreArray(vec, &array);
        PetscFunctionReturn(0);
    }

    // Gaussian distribution centered at the middle of the grid
    PetscErrorCode gaussian(Vec &vec, PetscInt gridSize, PetscInt localStart, PetscInt localEnd) {
        PetscFunctionBegin;
        PetscScalar *array;
        VecGetArray(vec, &array);

        PetscReal mean = gridSize / 2.0;
        PetscReal stddev = gridSize / 10.0;

        for (PetscInt i = localStart; i < localEnd; ++i) {
            array[i - localStart] = mean + stddev * gaussian_dist(generator);
        }

        VecRestoreArray(vec, &array);
        PetscFunctionReturn(0);
    }

    // Clustered positions at the center
    PetscErrorCode clustered(Vec &vec, PetscInt gridSize, PetscInt localStart, PetscInt localEnd) {
        PetscFunctionBegin;
        PetscScalar *array;
        VecGetArray(vec, &array);

        PetscReal center = gridSize / 2.0;
        for (PetscInt i = localStart; i < localEnd; ++i) {
            array[i - localStart] = center;
        }

        VecRestoreArray(vec, &array);
        PetscFunctionReturn(0);
    }

    // Random sparse positions
    PetscErrorCode random_sparse(Vec &vec, PetscInt gridSize, PetscInt localStart, PetscInt localEnd) {
        PetscFunctionBegin;
        PetscScalar *array;
        VecGetArray(vec, &array);

        for (PetscInt i = localStart; i < localEnd; ++i) {
            array[i - localStart] = uniform_dist(generator) * gridSize;
        }

        VecRestoreArray(vec, &array);
        PetscFunctionReturn(0);
    }
};

#endif // INITIALIZATION_HPP