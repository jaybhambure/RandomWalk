#include <petsc.h>
#include <petscviewerhdf5.h>
#include <random>

int main(int argc, char **argv) {
    PetscErrorCode ierr;
    
    // Initialize PETSc
    ierr = PetscInitialize(&argc, &argv, (char*)0, "1D Random Walk with DMDA"); CHKERRQ(ierr);
    
    // Simulation parameters
    PetscInt grid_size = 100;
    PetscInt num_steps = 1000;
    PetscInt num_walkers = 1000;
    PetscInt dof = 1;
    PetscInt stencil_width = 1;
    
    // Create 1D DMDA (single process, periodic boundaries)
    DM da;
    ierr = DMDACreate1d(PETSC_COMM_SELF, DM_BOUNDARY_PERIODIC, grid_size, 
                        dof, stencil_width, NULL, &da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    
    // Create vector for walker density
    Vec walker_density;
    ierr = DMCreateGlobalVector(da, &walker_density); CHKERRQ(ierr);
    ierr = VecZeroEntries(walker_density); CHKERRQ(ierr);
    
    // Array to store walker positions
    std::vector<PetscInt> walker_positions(num_walkers);
    
    // Store trajectories for plotting
    std::vector<std::vector<PetscInt>> trajectories(num_walkers);
    
    // Initialize walker positions (spread them out)
    for (PetscInt w = 0; w < num_walkers; w++) {
        walker_positions[w] = (grid_size / num_walkers) * w + grid_size / (2 * num_walkers);
        trajectories[w].push_back(walker_positions[w]);  // Store initial position
        ierr = VecSetValue(walker_density, walker_positions[w], 1.0, ADD_VALUES); CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(walker_density); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(walker_density); CHKERRQ(ierr);
    
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> step_choice(0, 1);  // 0 or 1
    
    PetscPrintf(PETSC_COMM_SELF, "Starting 1D Random Walk (Multiple Walkers)\n");
    PetscPrintf(PETSC_COMM_SELF, "Grid size: %d\n", grid_size);
    PetscPrintf(PETSC_COMM_SELF, "Number of walkers: %d\n", num_walkers);
    PetscPrintf(PETSC_COMM_SELF, "Number of steps: %d\n", num_steps);
    PetscPrintf(PETSC_COMM_SELF, "Initial positions: ");
    for (PetscInt w = 0; w < num_walkers; w++) {
        PetscPrintf(PETSC_COMM_SELF, "%d ", walker_positions[w]);
    }
    PetscPrintf(PETSC_COMM_SELF, "\n");
    
    // Perform random walk
    for (PetscInt i = 0; i < num_steps; i++) {
        // Clear all current positions
        ierr = VecZeroEntries(walker_density); CHKERRQ(ierr);
        
        // Move each walker
        for (PetscInt w = 0; w < num_walkers; w++) {
            // Take a random step (-1 or +1)
            PetscInt random_step = (step_choice(gen) == 0) ? -1 : 1;
            walker_positions[w] += random_step;
            
            // Apply periodic boundary conditions
            walker_positions[w] = (walker_positions[w] + grid_size) % grid_size;
            
            // Store position in trajectory (every 10 steps to avoid too much data)
            if (i % 10 == 0) {
                trajectories[w].push_back(walker_positions[w]);
            }
            
            // Set new position
            ierr = VecSetValue(walker_density, walker_positions[w], 1.0, ADD_VALUES); CHKERRQ(ierr);
        }
        
        ierr = VecAssemblyBegin(walker_density); CHKERRQ(ierr);
        ierr = VecAssemblyEnd(walker_density); CHKERRQ(ierr);
        
        // Print progress every 100 steps
        if (i % 100 == 0) {
            PetscPrintf(PETSC_COMM_SELF, "Step %d: positions = ", i);
            for (PetscInt w = 0; w < num_walkers; w++) {
                PetscPrintf(PETSC_COMM_SELF, "%d ", walker_positions[w]);
            }
            PetscPrintf(PETSC_COMM_SELF, "\n");
            
            // Optionally view the grid state
            if (i == 0 || i == 500) {
                PetscPrintf(PETSC_COMM_SELF, "DMDA grid state at step %d:\n", i);
                ierr = VecView(walker_density, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);
            }
        }
    }
    
    PetscPrintf(PETSC_COMM_SELF, "Final positions: ");
    for (PetscInt w = 0; w < num_walkers; w++) {
        PetscPrintf(PETSC_COMM_SELF, "%d ", walker_positions[w]);
    }
    PetscPrintf(PETSC_COMM_SELF, "\n");
    
    // Save data to HDF5 file
    PetscViewer viewer;
    ierr = PetscViewerHDF5Open(PETSC_COMM_SELF, "randWalk1d_data.h5", FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
    
    // Save simulation parameters as attributes
    ierr = PetscViewerHDF5WriteAttribute(viewer, NULL, "grid_size", PETSC_INT, &grid_size); CHKERRQ(ierr);
    ierr = PetscViewerHDF5WriteAttribute(viewer, NULL, "num_walkers", PETSC_INT, &num_walkers); CHKERRQ(ierr);
    ierr = PetscViewerHDF5WriteAttribute(viewer, NULL, "num_steps", PETSC_INT, &num_steps); CHKERRQ(ierr);
    
    // Create and save final positions vector
    Vec final_pos_vec;
    ierr = VecCreate(PETSC_COMM_SELF, &final_pos_vec); CHKERRQ(ierr);
    ierr = VecSetSizes(final_pos_vec, PETSC_DECIDE, num_walkers); CHKERRQ(ierr);
    ierr = VecSetFromOptions(final_pos_vec); CHKERRQ(ierr);
    
    for (PetscInt w = 0; w < num_walkers; w++) {
        PetscReal pos = (PetscReal)walker_positions[w];
        ierr = VecSetValue(final_pos_vec, w, pos, INSERT_VALUES); CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(final_pos_vec); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(final_pos_vec); CHKERRQ(ierr);
    
    ierr = PetscObjectSetName((PetscObject)final_pos_vec, "final_positions"); CHKERRQ(ierr);
    ierr = VecView(final_pos_vec, viewer); CHKERRQ(ierr);
    
    // Create and save trajectories matrix (flattened)
    PetscInt traj_size = num_walkers * trajectories[0].size();
    Vec traj_vec;
    ierr = VecCreate(PETSC_COMM_SELF, &traj_vec); CHKERRQ(ierr);
    ierr = VecSetSizes(traj_vec, PETSC_DECIDE, traj_size); CHKERRQ(ierr);
    ierr = VecSetFromOptions(traj_vec); CHKERRQ(ierr);
    
    PetscInt idx = 0;
    for (PetscInt w = 0; w < num_walkers; w++) {
        for (size_t t = 0; t < trajectories[w].size(); t++) {
            PetscReal pos = (PetscReal)trajectories[w][t];
            ierr = VecSetValue(traj_vec, idx++, pos, INSERT_VALUES); CHKERRQ(ierr);
        }
    }
    ierr = VecAssemblyBegin(traj_vec); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(traj_vec); CHKERRQ(ierr);
    
    ierr = PetscObjectSetName((PetscObject)traj_vec, "trajectories"); CHKERRQ(ierr);
    ierr = VecView(traj_vec, viewer); CHKERRQ(ierr);
    
    // Save trajectory dimensions as attributes
    PetscInt num_time_points = trajectories[0].size();
    ierr = PetscViewerHDF5WriteAttribute(viewer, NULL, "num_time_points", PETSC_INT, &num_time_points); CHKERRQ(ierr);
    
    // Clean up
    ierr = VecDestroy(&final_pos_vec); CHKERRQ(ierr);
    ierr = VecDestroy(&traj_vec); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    
    PetscPrintf(PETSC_COMM_SELF, "Data saved to randWalk1d_data.h5\n");
    PetscPrintf(PETSC_COMM_SELF, "Final DMDA grid state:\n");
    ierr = VecView(walker_density, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);
    
    // Clean up PETSc objects
    ierr = VecDestroy(&walker_density); CHKERRQ(ierr);
    ierr = DMDestroy(&da); CHKERRQ(ierr);
    
    // Finalize PETSc
    ierr = PetscFinalize(); CHKERRQ(ierr);
    return 0;
}
