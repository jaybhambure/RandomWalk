#!/usr/bin/env python3
"""
Visualization script for PETSc DMDA-based 1D random walk results
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_petsc_data(filename):
    """Load PETSc DMDA 1D random walk data from HDF5 file"""
    with h5py.File(filename, 'r') as f:
        x_coords = f['x_coordinates'][:]
        grid_pos = f['grid_positions'][:]
        grid_coords = f['grid_coordinates'][:]
        
        # Load attributes
        grid_size = f.attrs['grid_size']
        num_walkers = f.attrs['num_walkers']
        num_steps = f.attrs['num_steps']
        output_freq = f.attrs['output_frequency']
        domain_length = f.attrs['domain_length']
        
    num_outputs = num_steps // output_freq + 1
    
    # Reshape data: each walker's trajectory
    x_data = x_coords.reshape(num_walkers, num_outputs)
    grid_data = grid_pos.reshape(num_walkers, num_outputs).astype(int)
    
    return x_data, grid_data, grid_coords, num_walkers, num_outputs, grid_size, domain_length

def plot_petsc_trajectories(x_data, grid_coords, num_show=10):
    """Plot trajectories with PETSc grid overlay"""
    plt.figure(figsize=(14, 8))
    
    num_walkers = min(num_show, x_data.shape[0])
    colors = plt.cm.tab10(np.linspace(0, 1, num_walkers))
    
    for i in range(num_walkers):
        time_points = np.arange(x_data.shape[1])
        plt.plot(time_points, x_data[i, :], 'o-', 
                color=colors[i], alpha=0.7, markersize=4, 
                label=f'Walker {i}' if i < 5 else '')
    
    # Add grid lines at physical coordinates
    for i in range(0, len(grid_coords), 10):  # Show every 10th grid line
        plt.axhline(y=grid_coords[i], color='gray', alpha=0.3, linestyle='--')
    
    plt.xlabel('Time Step')
    plt.ylabel('X Position')
    plt.title(f'Trajectories on PETSc DMDA Grid ({num_walkers} walkers)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def plot_petsc_distribution_comparison(x_data, grid_data, grid_coords, grid_size):
    """Compare final distribution with PETSc grid structure"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Physical coordinates histogram
    final_x = x_data[:, -1]
    ax1.hist(final_x, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    ax1.set_xlabel('Final X Position')
    ax1.set_ylabel('Number of Walkers')
    ax1.set_title('Final Position Distribution (Physical Coordinates)')
    ax1.grid(True, alpha=0.3)
    
    # Distribution evolution over time
    time_points = np.arange(x_data.shape[1])
    mean_positions = np.mean(x_data, axis=0)
    std_positions = np.std(x_data, axis=0)
    
    ax2.plot(time_points, mean_positions, 'bo-', markersize=4, label='Mean Position')
    ax2.fill_between(time_points, 
                    mean_positions - std_positions,
                    mean_positions + std_positions,
                    alpha=0.3, label='±1 Std Dev')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Position')
    ax2.set_title('Position Statistics Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_petsc_diffusion(x_data, grid_coords):
    """Analyze diffusion on PETSc grid"""
    num_walkers, num_time_points = x_data.shape
    
    # Calculate mean squared displacement accounting for periodic boundaries
    initial_pos = x_data[:, 0]
    msd = np.zeros(num_time_points)
    domain_length = np.max(grid_coords) - np.min(grid_coords)
    
    for t in range(num_time_points):
        displacements = x_data[:, t] - initial_pos
        
        # Handle periodic boundary wrapping
        displacements = np.where(displacements > domain_length/2, 
                                displacements - domain_length, displacements)
        displacements = np.where(displacements < -domain_length/2, 
                                displacements + domain_length, displacements)
        msd[t] = np.mean(displacements**2)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # MSD vs time
    time_steps = np.arange(num_time_points)
    ax1.plot(time_steps, msd, 'bo-', markersize=4)
    
    # Fit linear relationship for diffusion analysis
    if num_time_points > 10:
        fit_start = max(5, num_time_points // 4)
        p = np.polyfit(time_steps[fit_start:], msd[fit_start:], 1)
        ax1.plot(time_steps[fit_start:], np.polyval(p, time_steps[fit_start:]), 
                'r--', label=f'Linear fit: D = {p[0]/2:.4f}')
        ax1.legend()
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Mean Squared Displacement')
    ax1.set_title('Diffusion Analysis on PETSc Grid')
    ax1.grid(True, alpha=0.3)
    
    # Displacement distribution at final time
    final_displacements = x_data[:, -1] - initial_pos
    final_displacements = np.where(final_displacements > domain_length/2, 
                                  final_displacements - domain_length, final_displacements)
    final_displacements = np.where(final_displacements < -domain_length/2, 
                                  final_displacements + domain_length, final_displacements)
    
    ax2.hist(final_displacements, bins=30, alpha=0.7, density=True, edgecolor='black')
    ax2.set_xlabel('Final Displacement')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Final Displacement Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Overlay Gaussian fit
    x_range = np.linspace(np.min(final_displacements), np.max(final_displacements), 100)
    variance = np.var(final_displacements)
    gaussian = np.exp(-x_range**2 / (2 * variance)) / np.sqrt(2 * np.pi * variance)
    ax2.plot(x_range, gaussian, 'r-', label=f'Gaussian (σ²={variance:.3f})')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    return msd

def main():
    parser = argparse.ArgumentParser(description='Visualize PETSc DMDA 1D Random Walk Data')
    parser.add_argument('filename', default='1drw_data.h5', nargs='?',
                       help='HDF5 file containing PETSc random walk data')
    parser.add_argument('--plot', choices=['trajectories', 'distribution', 'diffusion', 'all'],
                       default='all', help='Type of plot to generate')
    
    args = parser.parse_args()
    
    try:
        print(f"Loading PETSc DMDA data from {args.filename}...")
        x_data, grid_data, grid_coords, num_walkers, num_outputs, grid_size, domain_length = load_petsc_data(args.filename)
        
        print(f"Data loaded: {num_walkers} walkers, {num_outputs} time points")
        print(f"PETSc Grid: {grid_size} points, Domain: [0, {domain_length}]")
        print(f"Grid spacing: {np.mean(np.diff(grid_coords)):.6f}")
        
        if args.plot == 'trajectories' or args.plot == 'all':
            plot_petsc_trajectories(x_data, grid_coords, num_show=10)
        
        if args.plot == 'distribution' or args.plot == 'all':
            plot_petsc_distribution_comparison(x_data, grid_data, grid_coords, grid_size)
            
        if args.plot == 'diffusion' or args.plot == 'all':
            msd = analyze_petsc_diffusion(x_data, grid_coords)
            
    except FileNotFoundError:
        print(f"Error: File {args.filename} not found")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()