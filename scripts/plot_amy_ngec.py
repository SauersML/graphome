#!/usr/bin/env python3
"""
Plot NGEC values across sliding windows in the AMY gene cluster region.
Annotates known AMY gene locations.
"""

import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def main():
    if len(sys.argv) != 3:
        print("Usage: plot_amy_ngec.py <input_results.txt> <output_plot.png>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Read results: window_num start end ngec nodes edges
    windows = []
    ngec_values = []
    
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                window_num = int(parts[0])
                start = int(parts[1])
                end = int(parts[2])
                ngec = float(parts[3])
                windows.append((start, end))
                ngec_values.append(ngec)
    
    if not windows:
        print("No data found in input file")
        sys.exit(1)
    
    # Calculate window midpoints for x-axis
    midpoints = [(start + end) / 2 for start, end in windows]
    
    # AMY gene annotations (GRCh38 coordinates)
    genes = {
        'AMY2B': (103554644, 103579534, 'pancreatic'),
        'AMY2A': (103616651, 103625780, 'pancreatic'),
        'AMY1A': (103655519, 103664554, 'salivary'),
        'AMY1B': (103687415, 103696453, 'salivary (-)'),
        'AMY2Ap': (103713720, 103719905, 'pseudogene'),
        'AMY1C': (103749654, 103758692, 'salivary'),
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot NGEC values
    ax.plot(midpoints, ngec_values, 'o-', linewidth=2, markersize=4, 
            color='#2E86AB', label='NGEC', zorder=3)
    
    # Add gene annotations as colored spans
    colors = {
        'pancreatic': '#E63946',
        'salivary': '#06A77D',
        'salivary (-)': '#06A77D',
        'pseudogene': '#F77F00',
    }
    
    y_min, y_max = ax.get_ylim()
    
    for gene_name, (start, end, gene_type) in genes.items():
        color = colors.get(gene_type, '#999999')
        # Add vertical span
        ax.axvspan(start, end, alpha=0.2, color=color, zorder=1)
        # Add gene label
        mid = (start + end) / 2
        ax.text(mid, y_max * 0.95, gene_name, 
                rotation=90, va='top', ha='center',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
    
    # Formatting
    ax.set_xlabel('Genomic Position (chr1)', fontsize=12, fontweight='bold')
    ax.set_ylabel('NGEC (Normalized Graph Eigen-Complexity)', fontsize=12, fontweight='bold')
    ax.set_title('NGEC Across AMY Gene Cluster (5kb sliding windows)\nchr1:103,554,644-103,758,692', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Format x-axis with commas
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    
    # Legend for gene types
    legend_elements = [
        mpatches.Patch(color=colors['pancreatic'], alpha=0.5, label='Pancreatic amylase'),
        mpatches.Patch(color=colors['salivary'], alpha=0.5, label='Salivary amylase'),
        mpatches.Patch(color=colors['pseudogene'], alpha=0.5, label='Pseudogene'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # Add statistics
    stats_text = f'Windows: {len(windows)}\n'
    stats_text += f'Mean NGEC: {sum(ngec_values)/len(ngec_values):.4f}\n'
    stats_text += f'Min NGEC: {min(ngec_values):.4f}\n'
    stats_text += f'Max NGEC: {max(ngec_values):.4f}'
    
    ax.text(0.98, 0.02, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Windows analyzed: {len(windows)}")
    print(f"  Region: chr1:{windows[0][0]:,}-{windows[-1][1]:,}")
    print(f"  Mean NGEC: {sum(ngec_values)/len(ngec_values):.4f}")
    print(f"  NGEC range: [{min(ngec_values):.4f}, {max(ngec_values):.4f}]")

if __name__ == '__main__':
    main()
