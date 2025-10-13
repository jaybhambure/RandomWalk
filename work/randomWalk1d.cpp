#include <petsc.h>
#include <petscdmda.h>
#include <petscviewerhdf5.h>
#include "randomWalk1d.hpp"

int main(int argc, char **argv) {
    PetscInitialize(&argc, &argv, NULL, NULL);
    
    // Simulation parameters
    PetscInt GRID_SIZE = 100;           // Number of grid points
    PetscReal XMAX = 10.0;     // Physical domain: [0, 10]
    PetscInt NUM_WALKERS = 100;         // Number of random walkers
    PetscInt NUM_STEPS = 100;          // Steps per walker
    PetscInt OUTPUT_FREQUENCY = 50;     // Save every N steps
    
    // Create 1D DMDA grid with periodic boundaries
    DM dm;
    Vec coordinates;
    PetscInt dof = 1;              // Degrees of freedom per node
    PetscInt stencil_width = 1;    // Stencil width
    
    DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, GRID_SIZE, 
                 dof, stencil_width, NULL, &dm);
    DMSetFromOptions(dm);
    DMSetUp(dm);
    
    // Set uniform coordinates for the 1D grid and store them in the 'coordinates' vector 
    DMDASetUniformCoordinates(dm, 0.0, XMAX, 0.0, 0.0, 0.0, 0.0);
    DMGetCoordinates(dm, &coordinates);
    
    // Get grid information
    PetscReal grid_spacing = XMAX / (GRID_SIZE - 1);
    
    // Initialize random number generator
    GridRandomWalk1D rw(42);
    
    // Create PETSc vectors for walker positions
    Vec walker_positions, walker_coords;
    VecCreate(PETSC_COMM_WORLD, &walker_positions);
    VecSetSizes(walker_positions, PETSC_DECIDE, NUM_WALKERS);
    VecSetFromOptions(walker_positions);
    VecDuplicate(walker_positions, &walker_coords);
    
    // Initialize all walkers at center of domain
    PetscInt center_index = GRID_SIZE / 2;
    PetscReal center_x = center_index * grid_spacing;
    
    for (PetscInt w = 0; w < NUM_WALKERS; w++) {
        VecSetValue(walker_positions, w, (PetscScalar)center_index, INSERT_VALUES);
        VecSetValue(walker_coords, w, center_x, INSERT_VALUES);
    }
    VecAssemblyBegin(walker_positions); VecAssemblyEnd(walker_positions);
    VecAssemblyBegin(walker_coords); VecAssemblyEnd(walker_coords);
    
    // Storage for trajectories using PETSc vectors
    PetscInt num_outputs = NUM_STEPS / OUTPUT_FREQUENCY + 1;
    Vec trajectory_coords, trajectory_positions;
    VecCreate(PETSC_COMM_WORLD, &trajectory_coords);
    VecSetSizes(trajectory_coords, PETSC_DECIDE, NUM_WALKERS * num_outputs);
    VecSetFromOptions(trajectory_coords);
    VecDuplicate(trajectory_coords, &trajectory_positions);
    
    PetscInt output_count = 0;
    
    // Save initial positions to trajectory vectors
    PetscScalar *pos_array, *coord_array_walkers;
    VecGetArray(walker_positions, &pos_array);
    VecGetArray(walker_coords, &coord_array_walkers);
    
    for (PetscInt w = 0; w < NUM_WALKERS; w++) {
        PetscInt traj_index = w * num_outputs + output_count;
        VecSetValue(trajectory_positions, traj_index, pos_array[w], INSERT_VALUES);
        VecSetValue(trajectory_coords, traj_index, coord_array_walkers[w], INSERT_VALUES);
    }
    VecRestoreArray(walker_positions, &pos_array);
    VecRestoreArray(walker_coords, &coord_array_walkers);
    
    VecAssemblyBegin(trajectory_positions); VecAssemblyEnd(trajectory_positions);
    VecAssemblyBegin(trajectory_coords); VecAssemblyEnd(trajectory_coords);
    output_count++;
    
    // Main simulation loop
    
    for (PetscInt step = 0; step < NUM_STEPS; step++) {
        // Get current walker positions
        VecGetArray(walker_positions, &pos_array);
        VecGetArray(walker_coords, &coord_array_walkers);
        
        // Move all walkers simultaneously
        for (PetscInt w = 0; w < NUM_WALKERS; w++) {
            // Get random step (-1 or +1)
            PetscInt grid_step = rw.getRandomStep();
            
            // Update grid position with periodic boundary conditions
            PetscInt current_pos = (PetscInt)pos_array[w];
            PetscInt new_pos = (current_pos + grid_step + GRID_SIZE) % GRID_SIZE;
            
            // Update position and coordinate
            pos_array[w] = (PetscScalar)new_pos;
            coord_array_walkers[w] = new_pos * grid_spacing;
        }
        
        VecRestoreArray(walker_positions, &pos_array);
        VecRestoreArray(walker_coords, &coord_array_walkers);
        
        // Save positions at specified frequency
        if ((step + 1) % OUTPUT_FREQUENCY == 0) {
            VecGetArray(walker_positions, &pos_array);
            VecGetArray(walker_coords, &coord_array_walkers);
            
            for (PetscInt w = 0; w < NUM_WALKERS; w++) {
                PetscInt traj_index = w * num_outputs + output_count;
                VecSetValue(trajectory_positions, traj_index, pos_array[w], INSERT_VALUES);
                VecSetValue(trajectory_coords, traj_index, coord_array_walkers[w], INSERT_VALUES);
            }
            
            VecRestoreArray(walker_positions, &pos_array);
            VecRestoreArray(walker_coords, &coord_array_walkers);
            
            VecAssemblyBegin(trajectory_positions); VecAssemblyEnd(trajectory_positions);
            VecAssemblyBegin(trajectory_coords); VecAssemblyEnd(trajectory_coords);
            output_count++;
        }
    }
    
    // Calculate and display final distribution
    Vec grid_counts;
    VecCreateSeq(PETSC_COMM_SELF, GRID_SIZE, &grid_counts);
    VecZeroEntries(grid_counts);
    
    VecGetArray(walker_positions, &pos_array);
    for (PetscInt w = 0; w < NUM_WALKERS; w++) {
        PetscInt grid_pos = (PetscInt)pos_array[w];
        VecSetValue(grid_counts, grid_pos, 1.0, ADD_VALUES);
    }
    VecRestoreArray(walker_positions, &pos_array);
    VecAssemblyBegin(grid_counts); VecAssemblyEnd(grid_counts);
    
    // Write results to HDF5 file
    PetscViewer viewer;
    PetscViewerHDF5Open(PETSC_COMM_WORLD, "1drw_data.h5", FILE_MODE_WRITE, &viewer);
    
    // Write trajectory data
    PetscObjectSetName((PetscObject)trajectory_coords, "x_coordinates");
    PetscObjectSetName((PetscObject)trajectory_positions, "grid_positions");
    PetscObjectSetName((PetscObject)coordinates, "grid_coordinates");
    VecView(trajectory_coords, viewer);
    VecView(trajectory_positions, viewer);
    VecView(coordinates, viewer);
    
    // Write simulation parameters as attributes
    PetscViewerHDF5WriteAttribute(viewer, NULL, "grid_size", PETSC_INT, &GRID_SIZE);
    PetscViewerHDF5WriteAttribute(viewer, NULL, "num_walkers", PETSC_INT, &NUM_WALKERS);
    PetscViewerHDF5WriteAttribute(viewer, NULL, "num_steps", PETSC_INT, &NUM_STEPS);
    PetscViewerHDF5WriteAttribute(viewer, NULL, "output_frequency", PETSC_INT, &OUTPUT_FREQUENCY);
    PetscViewerHDF5WriteAttribute(viewer, NULL, "domain_length", PETSC_DOUBLE, &XMAX);
    
    // Cleanup
    VecDestroy(&walker_positions);
    VecDestroy(&walker_coords);
    VecDestroy(&trajectory_coords);
    VecDestroy(&trajectory_positions);
    VecDestroy(&grid_counts);
    DMDestroy(&dm);
    PetscViewerDestroy(&viewer);
    PetscFinalize();
    return 0;
}