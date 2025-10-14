#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import h5py
from collections import Counter

def plot_random_walk_results():
    """Plot displacement and final position histogram from HDF5 data"""
    
    # Read data from HDF5 file
    try:
        with h5py.File('randWalk1d_data.h5', 'r') as f:
            # Read attributes
            grid_size = int(f.attrs['grid_size'])
            num_walkers = int(f.attrs['num_walkers'])
            num_steps = int(f.attrs['num_steps'])
            num_time_points = int(f.attrs['num_time_points'])
            
            # Read datasets
            final_positions = f['final_positions'][:]
            trajectories_flat = f['trajectories'][:]
            
            print(f"Loaded data: {num_walkers} walkers, {num_time_points} time points")
            print(f"Grid size: {grid_size}, Total steps: {num_steps}")
            
    except FileNotFoundError:
        print("Error: randWalk1d_data.h5 not found. Run the C++ simulation first.")
        return
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")
        return
    
    # Reshape trajectories from flat array to matrix
    trajectories = trajectories_flat.reshape(num_walkers, num_time_points)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('1D Random Walk Analysis', fontsize=16)
    
    # 1. Final displacement histogram
    # Calculate final displacements for all walkers
    final_displacements = []
    for i in range(num_walkers):
        traj = trajectories[i]
        initial = traj[0]
        final = traj[-1]
        # Handle periodic boundaries
        diff = final - initial
        if diff > grid_size // 2:
            diff -= grid_size
        elif diff < -grid_size // 2:
            diff += grid_size
        final_displacements.append(diff)
    
    ax1.hist(final_displacements, bins=20, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Final Displacement', fontsize=12)
    ax1.set_ylabel('Number of Walkers', fontsize=12)
    ax1.set_title('Final Displacement Distribution', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No displacement')
    
    # 2. Displacement from starting positions
    time_points = np.arange(num_time_points) * 10  # Since we saved every 10 steps
    colors = plt.cm.tab10(np.linspace(0, 1, num_walkers))
    
    for i in range(num_walkers):
        traj = trajectories[i]
        initial_pos = traj[0]
        
        # Calculate displacement (handling periodic boundaries)
        displacement = []
        for pos in traj:
            # Calculate shortest distance on periodic grid
            diff = pos - initial_pos
            if diff > grid_size // 2:
                diff -= grid_size
            elif diff < -grid_size // 2:
                diff += grid_size
            displacement.append(diff)
        
        ax2.plot(time_points, displacement, '-', linewidth=1, 
                color=colors[i % len(colors)], alpha=0.6)
    
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Displacement from Start', fontsize=12)
    ax2.set_title('Displacement from Starting Position', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Show plot (no file saving)
    plt.show()
    
    # Print statistics
    print("\n=== Random Walk Statistics ===")
    print(f"Number of walkers: {num_walkers}")
    print(f"Grid size: {grid_size}")
    print(f"Total simulation steps: {num_steps}")
    print(f"Time points recorded: {num_time_points}")
    print(f"Final positions: {sorted(final_positions.astype(int))}")
    
    # Final displacements already calculated above
    
    print(f"Final displacements: {final_displacements}")
    print(f"Average displacement: {np.mean(final_displacements):.2f}")
    print(f"RMS displacement: {np.sqrt(np.mean(np.array(final_displacements)**2)):.2f}")

if __name__ == "__main__":
    plot_random_walk_results()