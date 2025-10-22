/* The grid size is 100 times the mean squared displacement (MSD) of the random walk
    to ensure various boundary conditions can be employed.
    Recall: For a 1D random walk with N steps
    - The expected displacement is zero.
    - The MSD is N, so the standard deviation is sqrt(N).
    - The probability distribution of walker positions after N steps approaches a Gaussian:
         P(x, N) ≈ (1 / sqrt(2 * pi * N)) * exp(- (x - x0)^2 / (2 * N))
    - For multiple walkers, the histogram of final positions should be approximately Gaussian,
      centered at the initial position (here, grid_size/2), with width ~ sqrt(num_steps).
    - The average displacement from the center should be close to zero,
      and the RMS displacement should be close to sqrt(num_steps).
*/

#include <petsc.h>
#include <petscviewerhdf5.h>
#include <random>

int main(int argc, char **argv) {
    PetscInitialize(&argc, &argv, (char*)0, "1D Random Walk with DMgrid");
    
    // NOTE SELF: Take simulation parameters from json
    PetscInt grid_size = 10000;      // Very large grid to minimize boundary effects
    PetscInt num_steps = 10000;
    PetscInt num_walkers = 1000;
    PetscInt dof = 1;
    PetscInt stencil_width = 1;
    
    // Save and print simulation parameters as attributes
    PetscViewer viewer;
    PetscViewerHDF5Open(PETSC_COMM_SELF, "randWalk1d_data.h5", FILE_MODE_WRITE, &viewer);
    PetscViewerHDF5WriteAttribute(viewer, NULL, "grid_size", PETSC_INT, &grid_size);
    PetscViewerHDF5WriteAttribute(viewer, NULL, "num_walkers", PETSC_INT, &num_walkers);
    PetscViewerHDF5WriteAttribute(viewer, NULL, "num_steps", PETSC_INT, &num_steps);

    PetscPrintf(PETSC_COMM_SELF, "Starting 1D Random Walk (Multiple Walkers)\n");
    PetscPrintf(PETSC_COMM_SELF, "Grid size: %d (center at %d)\n", grid_size, grid_size/2);
    PetscPrintf(PETSC_COMM_SELF, "Number of walkers: %d\n", num_walkers);
    PetscPrintf(PETSC_COMM_SELF, "Number of steps: %d\n", num_steps);
    PetscPrintf(PETSC_COMM_SELF, "Expected RMS displacement: ~%.1f\n", sqrt((double)num_steps));

    // Create 1D DMgrid (single process, no special boundary treatment)
    DM grid;
    DMDACreate1d(PETSC_COMM_SELF, DM_BOUNDARY_NONE, grid_size, dof, stencil_width, NULL, &grid);
    DMSetFromOptions(grid);
    DMSetUp(grid);

    // Create vector for walker density = no. of walkers/grid point
    Vec walker_density;
    DMCreateGlobalVector(grid, &walker_density);
    VecZeroEntries(walker_density);
    
    // PETSc Vec to store all walkers' current positions
    Vec allWalkers_current_pos;
    VecCreate(PETSC_COMM_SELF, &allWalkers_current_pos);
    VecSetSizes(allWalkers_current_pos, PETSC_DECIDE, num_walkers);
    VecSetFromOptions(allWalkers_current_pos);
    
    // Store trajectories for plotting (flattened PETSc Vec)
    PetscInt num_time_points = (num_steps / 10) + 1;  // Save every 10 steps + initial
    Vec walker_trajectories;
    VecCreate(PETSC_COMM_SELF, &walker_trajectories);
    VecSetSizes(walker_trajectories, PETSC_DECIDE, num_walkers * num_time_points);
    VecSetFromOptions(walker_trajectories);

    // Initialize walker positions (all start at center for Gaussian)
    PetscReal center_pos = (PetscReal)(grid_size / 2);
    for (PetscInt w = 0; w < num_walkers; w++) {
        VecSetValue(allWalkers_current_pos, w, center_pos, INSERT_VALUES);  // All walkers start at center
        // Store initial position in trajectory: index = w * num_time_points + 0
        PetscInt traj_idx = w * num_time_points + 0;
        VecSetValue(walker_trajectories, traj_idx, center_pos, INSERT_VALUES);
        VecSetValue(walker_density, (PetscInt)center_pos, 1.0, ADD_VALUES);
    }   
    VecAssemblyBegin(allWalkers_current_pos);
    VecAssemblyEnd(allWalkers_current_pos);
    VecAssemblyBegin(walker_density);
    VecAssemblyEnd(walker_density);
    
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> step_choice(0, 1);  // 0 or 1
    
    // Perform random walk
    for (PetscInt i = 0; i < num_steps; i++) {
        
        VecZeroEntries(walker_density);
        
        // Move each walker
        for (PetscInt w = 0; w < num_walkers; w++) {
            // Get current position
            PetscReal current_pos;
            VecGetValues(allWalkers_current_pos, 1, &w, &current_pos);
            
            // Take a random step (-1 or +1)
            PetscInt random_step = (step_choice(gen) == 0) ? -1 : 1;
            PetscReal new_pos = current_pos + (PetscReal)random_step;
            VecSetValue(allWalkers_current_pos, w, new_pos, INSERT_VALUES);
            
            // Store position in trajectory (every 10 steps to avoid too much data)
            if (i % 10 == 0) {
                PetscInt time_idx = (i / 10) + 1;  // +1 because initial position is at index 0
                PetscInt traj_idx = w * num_time_points + time_idx;
                VecSetValue(walker_trajectories, traj_idx, new_pos, INSERT_VALUES);
            }
            
            // Only add to grid density if walker is still in domain will be useful for other BCs
            if (new_pos >= 0 && new_pos < grid_size) {
                VecSetValue(walker_density, (PetscInt)new_pos, 1.0, ADD_VALUES);
            }
        }
        // Assemble walker positions after all updates
        VecAssemblyBegin(allWalkers_current_pos);
        VecAssemblyEnd(allWalkers_current_pos);
        
        VecAssemblyBegin(walker_density);
        VecAssemblyEnd(walker_density);
    }
    
    // Assemble the trajectory vector after all values are set
    VecAssemblyBegin(walker_trajectories);
    VecAssemblyEnd(walker_trajectories);
    
    // Save final positions (allWalkers_current_pos IS the final positions)
    PetscObjectSetName((PetscObject)allWalkers_current_pos, "final_positions");
    VecView(allWalkers_current_pos, viewer);
    
    // Save trajectories (already in PETSc Vec format)
    PetscObjectSetName((PetscObject)walker_trajectories, "trajectories");
    VecView(walker_trajectories, viewer);
    
    // Save trajectory dimensions as attributes
    PetscViewerHDF5WriteAttribute(viewer, NULL, "num_time_points", PETSC_INT, &num_time_points);
    
    // Calculate final statistics using PETSc functions
    PetscReal sum, min_pos, max_pos;
    PetscInt min_idx, max_idx;
    
    // Use PETSc built-in statistics functions
    VecSum(allWalkers_current_pos, &sum);
    VecMin(allWalkers_current_pos, &min_idx, &min_pos);
    VecMax(allWalkers_current_pos, &max_idx, &max_pos);
    
    PetscReal avg_pos = sum / num_walkers;
    
    // Count walkers still in grid bounds
    PetscInt walkers_in_grid = 0;
    for (PetscInt w = 0; w < num_walkers; w++) {
        PetscReal pos;
        VecGetValues(allWalkers_current_pos, 1, &w, &pos);
        if (pos >= 0 && pos < grid_size) walkers_in_grid++;
    }
    
    PetscReal avg_displacement = avg_pos - (grid_size / 2);
    
    PetscPrintf(PETSC_COMM_SELF, "\nFinal Results:\n");
    PetscPrintf(PETSC_COMM_SELF, "Average final position: %.2f (center is %d)\n", avg_pos, grid_size/2);
    PetscPrintf(PETSC_COMM_SELF, "Average displacement from center: %.2f\n", avg_displacement);
    PetscPrintf(PETSC_COMM_SELF, "Position range: [%d, %d]\n", (int)min_pos, (int)max_pos);
    PetscPrintf(PETSC_COMM_SELF, "Walkers still in grid: %d/%d\n", walkers_in_grid, num_walkers);
    
    // Clean up
    VecDestroy(&allWalkers_current_pos);
    VecDestroy(&walker_trajectories);
    PetscViewerDestroy(&viewer);
    
    // Clean up PETSc objects
    VecDestroy(&walker_density);
    DMDestroy(&grid);
    
    // Finalize PETSc
    PetscFinalize();
    return 0;
}
