#ifndef INITIALIZE_HPP
#define INITIALIZE_HPP

#include <petsc.h>
#include <random>

PetscErrorCode init_fine_grid_walkers(Vec fine_walker_positions, PetscInt fine_grid_size) {
    PetscReal *positions_array;

    VecGetArray(fine_walker_positions, &positions_array);

    // Random number generator for Gaussian distribution
    std::random_device rd;
    std::mt19937 gen(rd());
        
    // Initialize walker positions with Gaussian distribution on the fine grid
    PetscReal center_pos = (PetscReal)(fine_grid_size / 2);
    PetscReal sigma = (PetscReal)(fine_grid_size / 20);  // Standard deviation (adjust as needed)
    std::normal_distribution<PetscReal> gaussian_dist(center_pos, sigma);
        
    for (PetscInt w = 0; w < 1000; w++) {  // Assume 1000 walkers for example
        PetscReal pos = gaussian_dist(gen);
                
        // Handle periodic boundary conditions - wrap around the grid
        while (pos < 0) pos += fine_grid_size;
        while (pos >= fine_grid_size) pos -= fine_grid_size;
                
        positions_array[w] = pos;
    }

    VecRestoreArray(fine_walker_positions, &positions_array);
    VecAssemblyBegin(fine_walker_positions);
    VecAssemblyEnd(fine_walker_positions);
        
    return 0;
}

#endif // INITIALIZE_HPP