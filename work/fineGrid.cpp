#include <petsc.h>
#include "initialize.hpp"
#include <petscviewerhdf5.h>
#include <random>

int main(int argc, char** argv){
    PetscInitialize(&argc, &argv, (char*)0, "Fine-Coarse Grid Transfer Demo");

    // fine grid parameters
    DM fine_grid;
    PetscInt fine_grid_size = 100;
    PetscInt dof = 1;
    PetscInt stencil_width = 1;

    // setup fine grid with periodic boundaries
    DMDACreate1d(PETSC_COMM_SELF, DM_BOUNDARY_PERIODIC, fine_grid_size, 
                    dof, stencil_width, NULL, &fine_grid);
    DMSetFromOptions(fine_grid);
    DMSetUp(fine_grid);

    // initialize walker positions (1000 walkers)
    PetscInt num_walkers = 1000;
    Vec fine_walker_positions;
    VecCreate(PETSC_COMM_SELF, &fine_walker_positions);
    VecSetSizes(fine_walker_positions, PETSC_DECIDE, num_walkers);
    VecSetFromOptions(fine_walker_positions);
    // initialization function
    init_fine_grid_walkers(fine_walker_positions, fine_grid_size);

    // save this fine grid walker positions to file
    PetscViewer viewer;
    PetscViewerHDF5Open(PETSC_COMM_SELF, "fine_grid_walkers.h5", FILE_MODE_WRITE, &viewer);
    PetscObjectSetName((PetscObject)fine_walker_positions, "fine_walker_positions");
    VecView(fine_walker_positions, viewer);

    // coarse grid parameters
    PetscInt coarse_grid_size = fine_grid_size / 2;
    DM coarse_grid;
    
    // setup coarse grid
    DMDACreate1d(PETSC_COMM_SELF, DM_BOUNDARY_PERIODIC, coarse_grid_size, 
                    dof, stencil_width, NULL, &coarse_grid);
    DMSetFromOptions(coarse_grid);
    DMSetUp(coarse_grid);

    // setup coarse grid vector
    Vec coarse_walker_positions;
    VecCreate(PETSC_COMM_SELF, &coarse_walker_positions);
    VecSetSizes(coarse_walker_positions, PETSC_DECIDE, num_walkers);
    VecSetFromOptions(coarse_walker_positions);
    
    // STEP 1: Create fine grid density from walker positions
    Vec fine_density;
    DMCreateGlobalVector(fine_grid, &fine_density);
    VecZeroEntries(fine_density);
    
    // Count walkers in each fine grid cell
    PetscReal *fine_positions_array;
    VecGetArray(fine_walker_positions, &fine_positions_array);
    
    for (PetscInt w = 0; w < num_walkers; w++) {
        PetscInt cell = (PetscInt)fine_positions_array[w];  // Grid cell containing walker
        if (cell >= 0 && cell < fine_grid_size) {  // Safety check
            VecSetValue(fine_density, cell, 1.0, ADD_VALUES);  // Count walker in this cell
        }
    }
    VecAssemblyBegin(fine_density);
    VecAssemblyEnd(fine_density);
    
    // STEP 2: Coarsen the density (average pairs of fine cells -> coarse cells)
    Vec coarse_density;
    DMCreateGlobalVector(coarse_grid, &coarse_density);
    VecZeroEntries(coarse_density);
    
    PetscReal *fine_density_array, *coarse_density_array;
    VecGetArray(fine_density, &fine_density_array);
    VecGetArray(coarse_density, &coarse_density_array);
    
    for (PetscInt c = 0; c < coarse_grid_size; c++) {
        // Each coarse cell averages two fine cells
        PetscInt fine_idx1 = c * 2;
        PetscInt fine_idx2 = fine_idx1 + 1;
        
        // Handle periodic boundaries
        if (fine_idx2 >= fine_grid_size) fine_idx2 = 0;
        
        coarse_density_array[c] = 0.5 * (fine_density_array[fine_idx1] + fine_density_array[fine_idx2]);
    }
    
    VecRestoreArray(fine_density, &fine_density_array);
    VecRestoreArray(coarse_density, &coarse_density_array);
    VecAssemblyBegin(coarse_density);
    VecAssemblyEnd(coarse_density);
    
    // STEP 3: Map walker positions to coarse grid coordinates
    PetscReal *coarse_positions_array;
    VecGetArray(coarse_walker_positions, &coarse_positions_array);
    VecGetArray(coarse_walker_positions, &coarse_positions_array);
    
    // Map each walker position from fine grid to coarse grid
    for (PetscInt w = 0; w < num_walkers; w++) {
        PetscReal fine_pos = fine_positions_array[w];
        PetscReal coarse_pos = fine_pos / 2.0;  // Scale position to coarse grid
        
        // Handle periodic boundaries on coarse grid
        while (coarse_pos >= coarse_grid_size) coarse_pos -= coarse_grid_size;
        while (coarse_pos < 0) coarse_pos += coarse_grid_size;
        
        coarse_positions_array[w] = coarse_pos;
    }
    
    VecRestoreArray(fine_walker_positions, &fine_positions_array);
    VecRestoreArray(coarse_walker_positions, &coarse_positions_array);
    VecAssemblyBegin(coarse_walker_positions);
    VecAssemblyEnd(coarse_walker_positions);
    
    // Save coarse grid data
    PetscObjectSetName((PetscObject)coarse_density, "coarse_density");
    VecView(coarse_density, viewer);
    PetscObjectSetName((PetscObject)coarse_walker_positions, "coarse_walker_positions");
    VecView(coarse_walker_positions, viewer);
    
    // STEP 4: Perform random walk on coarse grid
    PetscInt num_steps = 1000;  // Number of random walk steps
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> step_choice(0, 1);  // 0 or 1 for -1 or +1 step
    
    PetscPrintf(PETSC_COMM_SELF, "Performing %d random walk steps on coarse grid...\n", num_steps);
    
    // Get array access for random walk
    VecGetArray(coarse_walker_positions, &coarse_positions_array);
    
    for (PetscInt step = 0; step < num_steps; step++) {
        for (PetscInt w = 0; w < num_walkers; w++) {
            PetscReal current_pos = coarse_positions_array[w];
            
            // Take random step (-1 or +1)
            PetscInt random_step = (step_choice(gen) == 0) ? -1 : 1;
            PetscReal new_pos = current_pos + (PetscReal)random_step;
            
            // Handle periodic boundaries on coarse grid
            while (new_pos >= coarse_grid_size) new_pos -= coarse_grid_size;
            while (new_pos < 0) new_pos += coarse_grid_size;
            
            coarse_positions_array[w] = new_pos;
        }
    }
    
    VecRestoreArray(coarse_walker_positions, &coarse_positions_array);
    VecAssemblyBegin(coarse_walker_positions);
    VecAssemblyEnd(coarse_walker_positions);
    
    // STEP 5: Map coarse results back to fine grid
    Vec refined_walker_positions;
    VecCreate(PETSC_COMM_SELF, &refined_walker_positions);
    VecSetSizes(refined_walker_positions, PETSC_DECIDE, num_walkers);
    VecSetFromOptions(refined_walker_positions);
    
    PetscReal *refined_positions_array;
    VecGetArray(coarse_walker_positions, &coarse_positions_array);
    VecGetArray(refined_walker_positions, &refined_positions_array);
    
    // Map each coarse position back to fine grid
    for (PetscInt w = 0; w < num_walkers; w++) {
        PetscReal coarse_pos = coarse_positions_array[w];
        PetscReal fine_pos = coarse_pos * 2.0;  // Scale back to fine grid
        
        // Handle periodic boundaries on fine grid
        while (fine_pos >= fine_grid_size) fine_pos -= fine_grid_size;
        while (fine_pos < 0) fine_pos += fine_grid_size;
        
        refined_positions_array[w] = fine_pos;
    }
    
    VecRestoreArray(coarse_walker_positions, &coarse_positions_array);
    VecRestoreArray(refined_walker_positions, &refined_positions_array);
    VecAssemblyBegin(refined_walker_positions);
    VecAssemblyEnd(refined_walker_positions);
    
    // Save final results
    PetscObjectSetName((PetscObject)coarse_walker_positions, "coarse_walker_final");
    VecView(coarse_walker_positions, viewer);
    PetscObjectSetName((PetscObject)refined_walker_positions, "refined_walker_final");
    VecView(refined_walker_positions, viewer);
    
    // Print statistics
    PetscReal coarse_sum, fine_sum;
    VecSum(coarse_walker_positions, &coarse_sum);
    VecSum(refined_walker_positions, &fine_sum);
    
    PetscPrintf(PETSC_COMM_SELF, "\nRandom Walk Results:\n");
    PetscPrintf(PETSC_COMM_SELF, "Coarse grid mean position: %.2f\n", coarse_sum / num_walkers);
    PetscPrintf(PETSC_COMM_SELF, "Refined fine grid mean position: %.2f\n", fine_sum / num_walkers);
    PetscPrintf(PETSC_COMM_SELF, "Expected scaling: refined ≈ 2 × coarse\n");


    // cleanup
    VecDestroy(&fine_walker_positions);
    VecDestroy(&coarse_walker_positions);
    VecDestroy(&refined_walker_positions);
    VecDestroy(&fine_density);
    VecDestroy(&coarse_density);
    DMDestroy(&fine_grid);
    DMDestroy(&coarse_grid);
    PetscViewerDestroy(&viewer);
    
    PetscFinalize();
    return 0;
}