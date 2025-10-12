#include "random_walker.hpp"
#include "initialization.hpp"
#include <petsc.h>
#include <petscviewerhdf5.h>

int main(int argc, char **argv) {
    PetscInitialize(&argc, &argv, NULL, NULL);

    PetscInt num_walkers = 1000; // Number of walkers
    PetscInt num_steps = 1000; // Number of steps per walker
    PetscInt gridSize = 100; // Define the grid size for initialization

    Vec final_positions;
    VecCreate(PETSC_COMM_WORLD, &final_positions);
    VecSetSizes(final_positions, PETSC_DECIDE, num_walkers);
    VecSetFromOptions(final_positions);

    InitialPositions initializer;

    Vec initial_positions;
    VecCreate(PETSC_COMM_WORLD, &initial_positions);
    VecSetSizes(initial_positions, PETSC_DECIDE, num_walkers);
    VecSetFromOptions(initial_positions);

    // Initialize the positions of walkers using a uniform distribution
    initializer.gaussian(initial_positions, gridSize);

    // Save initial positions to the HDF5 file
    PetscViewer viewer;
    PetscViewerHDF5Open(PETSC_COMM_WORLD, "walker_data.h5", FILE_MODE_WRITE, &viewer);
    PetscObjectSetName((PetscObject)initial_positions, "initial_positions");
    VecView(initial_positions, viewer);

    // Clean up initial positions vector
    VecDestroy(&initial_positions);

    for (PetscInt w = 0; w < num_walkers; ++w) {
        Vec walk;
        VecCreate(PETSC_COMM_WORLD, &walk);
        VecSetSizes(walk, PETSC_DECIDE, num_steps);
        VecSetFromOptions(walk);

        RandomWalker walker;

        // Set random steps in the vector
        walker.take_steps(walk, num_steps);

        // Calculate the final position of the walker
        PetscScalar final_position = 0;
        PetscScalar *walkArray;
        VecGetArray(walk, &walkArray);
        for (PetscInt i = 0; i < num_steps; ++i) {
            final_position += walkArray[i];
        }
        VecRestoreArray(walk, &walkArray);

        // Save the final position to the final_positions vector
        VecSetValue(final_positions, w, final_position, INSERT_VALUES);

        // Clean up
        VecDestroy(&walk);
    }

    VecAssemblyBegin(final_positions);
    VecAssemblyEnd(final_positions);

    // Save all final positions to an HDF5 file
    PetscObjectSetName((PetscObject)final_positions, "final_positions");
    VecView(final_positions, viewer);
    PetscViewerDestroy(&viewer);

    // Clean up
    VecDestroy(&final_positions);
    PetscFinalize();
    return 0;
}