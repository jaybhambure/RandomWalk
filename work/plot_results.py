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
    
    # Smart bin selection for optimal Gaussian visualization
    displacements_array = np.array(final_displacements)
    
    # Method 1: Scott's rule (optimal for Gaussian data)
    sigma = np.std(displacements_array)
    n = len(displacements_array)
    scott_width = 3.5 * sigma / (n**(1/3))
    scott_bins = int((np.max(displacements_array) - np.min(displacements_array)) / scott_width)
    
    # Method 2: Freedman-Diaconis rule (robust against outliers)
    q75, q25 = np.percentile(displacements_array, [75, 25])
    iqr = q75 - q25
    fd_width = 2 * iqr / (n**(1/3))
    fd_bins = int((np.max(displacements_array) - np.min(displacements_array)) / fd_width)
    
    # Method 3: Square root rule (simple and reliable)
    sqrt_bins = int(np.sqrt(n))
    
    # Choose the best method (Scott's rule for Gaussian data, but constrain to reasonable range)
    optimal_bins = max(20, min(50, scott_bins))  # Between 20-50 bins for good resolution
    
    print(f"Bin selection analysis:")
    print(f"  Scott's rule: {scott_bins} bins")
    print(f"  Freedman-Diaconis: {fd_bins} bins") 
    print(f"  Square root rule: {sqrt_bins} bins")
    print(f"  Using: {optimal_bins} bins")
    
    # Create histogram with optimal binning
    n_hist, bins_hist, patches = ax1.hist(final_displacements, bins=optimal_bins, 
                                         alpha=0.7, color='skyblue', edgecolor='navy',
                                         density=False)
    
    # Overlay theoretical Gaussian for comparison
    x_theory = np.linspace(np.min(final_displacements), np.max(final_displacements), 1000)
    mean_theory = np.mean(final_displacements)
    std_theory = np.std(final_displacements)
    gaussian_theory = n * (bins_hist[1] - bins_hist[0]) * (1/(std_theory * np.sqrt(2*np.pi))) * \
                     np.exp(-0.5 * ((x_theory - mean_theory) / std_theory)**2)
    
    ax1.plot(x_theory, gaussian_theory, 'r-', linewidth=1.5, 
             label=f'Theoretical Gaussian\n(μ={mean_theory:.1f}, σ={std_theory:.1f})')
    
    ax1.set_xlabel('Final Displacement', fontsize=12)
    ax1.set_ylabel('Number of Walkers', fontsize=12)
    ax1.set_title(f'Final Displacement Distribution ({optimal_bins} bins)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero displacement')
    ax1.legend()
    
    # 2. Displacement from starting positions
    time_points = np.arange(num_time_points) * 10  # Since we saved every 10 steps
    
    # Plot only a subset of walkers to reduce visual clutter
    num_plot_walkers = min(50, num_walkers)  # Plot max 50 walkers for clarity
    plot_indices = np.linspace(0, num_walkers-1, num_plot_walkers, dtype=int)
    colors = plt.cm.tab10(np.linspace(0, 1, num_plot_walkers))
    
    for idx, i in enumerate(plot_indices):
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
        
        ax2.plot(time_points, displacement, '-', linewidth=0.8, 
                color=colors[idx % len(colors)], alpha=0.7)
    
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Displacement from Start', fontsize=12)
    ax2.set_title('Displacement from Starting Position', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
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
    
    # Save plot and show once
    plt.savefig('random_walk_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_random_walk_results()