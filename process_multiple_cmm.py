import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict
import os
import glob
from tqdm import tqdm

# Define standard chromosome name list (without suffix)
STANDARD_CHRS = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY', 'chrM', 'chrMT']

def compute_radius(x, y, z, debug_info=""):
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)) or not isinstance(z, (int, float)):
        return np.nan
    try:
        dist_sq = x**2 + y**2 + z**2
        if dist_sq < 0:
            return np.nan
        return np.sqrt(dist_sq)
    except Exception as e:
        return np.nan

def is_valid_coordinate_string(coord_str):
    if coord_str.lower() == '-nan':
        return False
    try:
        float(coord_str)
        return True
    except ValueError:
        return False

def parse_cmm(cmm_file):
    regional_beads_raw = {}
    
    with open(cmm_file, 'r') as f:
        for line in f:
            if line.startswith('<marker'):
                chr_match = re.search(r'chrID="([^"]+)"', line)
                bead_match = re.search(r'beadID="([^"]+)"', line)
                x_match = re.search(r'x="([^"]+)"', line)
                y_match = re.search(r'y="([^"]+)"', line)
                z_match = re.search(r'z="([^"]+)"', line)
                if chr_match and bead_match and x_match and y_match and z_match:
                    chr_id_full = chr_match.group(1)
                    bead_id_full = bead_match.group(1)
                    x_str = x_match.group(1)
                    y_str = y_match.group(1)
                    z_str = z_match.group(1)
                    allele = None
                    chr_id_clean = chr_id_full
                    if chr_id_full.endswith('_A'):
                        allele = 'A'
                        chr_id_clean = chr_id_full[:-2]
                    elif chr_id_full.endswith('_B'):
                        allele = 'B'
                        chr_id_clean = chr_id_full[:-2]
                    elif chr_id_full in STANDARD_CHRS:
                        allele = 'None'
                        chr_id_clean = chr_id_full
                    if chr_id_clean not in STANDARD_CHRS or allele is None:
                        continue
                    pos_match = re.search(r':(\d+)-(\d+)', bead_id_full)
                    if pos_match:
                        start = int(pos_match.group(1))
                        end = int(pos_match.group(2))
                        region_key = (chr_id_clean, start, end)
                        if region_key not in regional_beads_raw:
                            regional_beads_raw[region_key] = {}
                        regional_beads_raw[region_key][allele] = {
                            'chr_id_full': chr_id_full,
                            'bead_id_full': bead_id_full,
                            'x_str': x_str,
                            'y_str': y_str,
                            'z_str': z_str
                        }
    final_beads = []
    for region_key, alleles_data in regional_beads_raw.items():
        chr_id_clean, start, end = region_key
        selected_bead_data = None
        if 'B' in alleles_data:
            b_data = alleles_data['B']
            if all(is_valid_coordinate_string(b_data[coord]) for coord in ['x_str', 'y_str', 'z_str']):
                selected_bead_data = b_data
        if selected_bead_data is None and 'A' in alleles_data:
            a_data = alleles_data['A']
            if all(is_valid_coordinate_string(a_data[coord]) for coord in ['x_str', 'y_str', 'z_str']):
                selected_bead_data = a_data
        if selected_bead_data is None and 'None' in alleles_data:
            none_data = alleles_data['None']
            if all(is_valid_coordinate_string(none_data[coord]) for coord in ['x_str', 'y_str', 'z_str']):
                selected_bead_data = none_data
        if selected_bead_data:
            try:
                x = float(selected_bead_data['x_str'])
                y = float(selected_bead_data['y_str'])
                z = float(selected_bead_data['z_str'])
                r = compute_radius(x, y, z)
                final_beads.append((
                    chr_id_clean,
                    start,
                    end,
                    x, y, z,
                    r
                ))
            except ValueError:
                continue
    return final_beads

def process_multiple_files(cmm_files):
    # Store beads data from all files
    all_beads_data = defaultdict(list)
    nuclear_radii = []  # Store nuclear radius for each model
    
    # Use tqdm to show progress bar
    for cmm_file in tqdm(cmm_files, desc="Processing files"):
        beads = parse_cmm(cmm_file)
        # Collect all valid radial distances for current model
        model_radii = []
        for bead in beads:
            chr_id, start, end = bead[0:3]
            r = bead[6]
            if not np.isnan(r):
                all_beads_data[(chr_id, start, end)].append(r)
                model_radii.append(r)
        
        # If current model has sufficient data, calculate its nuclear radius (mean of top 5% maximum distances)
        if len(model_radii) >= 20:  # Ensure at least 20 points before calculating top 5%
            # Sort by distance
            sorted_radii = sorted(model_radii, reverse=True)
            # Calculate number of top 5%
            top_5_percent = max(1, int(len(sorted_radii) * 0.05))
            # Use mean of top 5% as nuclear radius for this model
            model_nuclear_radius = np.mean(sorted_radii[:top_5_percent])
            nuclear_radii.append(model_nuclear_radius)
    
    # Calculate average nuclear radius
    if nuclear_radii:
        avg_nuclear_radius = np.mean(nuclear_radii)
        std_nuclear_radius = np.std(nuclear_radii)
        print(f"\nCalculated average nuclear radius: {avg_nuclear_radius:.2f} ± {std_nuclear_radius:.2f} μm")
        print(f"Nuclear radius range: {min(nuclear_radii):.2f} - {max(nuclear_radii):.2f} μm")
        print(f"Number of valid models: {len(nuclear_radii)}")
        print(f"Each model uses the mean of top 5% farthest distances as nuclear radius")
    else:
        avg_nuclear_radius = 5.0  # Default value
        print("\nWarning: Unable to calculate average nuclear radius, using default value 5.0 μm")
    
    # Calculate averages
    averaged_beads = []
    for (chr_id, start, end), radii in all_beads_data.items():
        if radii:  # Ensure valid data exists
            avg_r = np.mean(radii)
            averaged_beads.append((chr_id, start, end, avg_r))
    
    return averaged_beads, avg_nuclear_radius

def read_lad_bed(bed_file):
    """Read LAD BED file, return dictionary {(chr, start, end): True}"""
    lad_regions = {}
    with open(bed_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 3:
                    chr_id = parts[0]
                    start = int(parts[1])
                    end = int(parts[2])
                    lad_regions[(chr_id, start, end)] = True
    return lad_regions

def merge_overlapping_regions(regions):
    """Merge overlapping or adjacent regions"""
    if not regions:
        return []
    
    # Sort by chromosome and start position
    sorted_regions = sorted(regions, key=lambda x: (x[0], x[1]))
    merged = []
    current = list(sorted_regions[0])
    
    for region in sorted_regions[1:]:
        if region[0] == current[0] and region[1] <= current[2] + 1:  # Same chromosome and adjacent or overlapping
            current[2] = max(current[2], region[2])
        else:
            merged.append(tuple(current))
            current = list(region)
    
    merged.append(tuple(current))
    return merged

def plot_averaged_beads(averaged_beads, output_filename="averaged_heatmap.pdf", max_mb=250, max_radius=None, lad_regions=None, title=None):
    chr_order_base = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']
    actual_chrs = sorted(list(set([b[0] for b in averaged_beads])))
    
    chr_order = [c for c in chr_order_base if c in actual_chrs]
    for ac in actual_chrs:
        if ac not in chr_order:
            chr_order.append(ac)
    
    chr2idx = {c: i for i, c in enumerate(chr_order)}
    
    fig, ax = plt.subplots(figsize=(8, max(5, len(chr_order)*0.2)))
    
    all_r = [b[3] for b in averaged_beads if b[0] in chr2idx and not np.isnan(b[3])]
    
    if not all_r:
        vmin = 0
        vmax = 1
        cmap = plt.get_cmap('Spectral_r')
        color_fixed = True
        fixed_color = 'lightgray'
    else:
        vmin = 0  # Fixed minimum value at 0
        vmax = max_radius if max_radius is not None else max(all_r)
        cmap = plt.get_cmap('Spectral_r')
        color_fixed = False
        if vmin == vmax:
            color_fixed = True
            fixed_color = 'red'
    
    # First draw all rectangles
    for chr_id, start, end, r in averaged_beads:
        if chr_id not in chr2idx:
            continue
        y_idx = chr2idx[chr_id]
        
        if color_fixed:
            color = fixed_color
        elif np.isnan(r):
            color = 'gray'
        else:
            norm_r = (r - vmin) / (vmax - vmin)
            color = cmap(norm_r)
        
        start_mb = start/1e6
        end_mb = end/1e6
        
        if start_mb >= max_mb:
            continue
        
        rect = Rectangle((start_mb, y_idx-0.4), (end_mb - start_mb), 0.8, color=color, linewidth=0)
        ax.add_patch(rect)
    
    # Then draw merged LAD borders
    if lad_regions is not None:
        # Process LAD regions grouped by chromosome
        for chr_id in chr_order:
            if chr_id not in chr2idx:
                continue
            
            # Get all LAD regions for current chromosome
            chr_lad_regions = [(s, e) for c, s, e in lad_regions.keys() if c == chr_id]
            if not chr_lad_regions:
                continue
            
            # Merge adjacent or overlapping regions
            merged_regions = merge_overlapping_regions([(chr_id, s, e) for s, e in chr_lad_regions])
            
            # Draw merged borders
            y_idx = chr2idx[chr_id]
            for _, start, end in merged_regions:
                start_mb = start/1e6
                end_mb = end/1e6
                if start_mb >= max_mb:
                    continue
                
                border = Rectangle((start_mb, y_idx-0.4), (end_mb - start_mb), 0.8, 
                                 fill=False, edgecolor='black', linewidth=0.5)
                ax.add_patch(border)
    
    ax.set_xlim(0, max_mb)
    ax.set_ylim(-0.5, len(chr_order)-0.5)
    ax.set_yticks(np.arange(len(chr_order)))
    ax.set_yticklabels(chr_order)
    ax.set_xlabel('Chromosome position (megabase)')
    ax.set_ylabel('Chromosomes')
    
    if not color_fixed and vmin != vmax:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_ticks([vmin, vmax/2, vmax])
        if max_radius == 1.0:  # Relative distance case
            cbar.set_ticklabels(['Nuclear center (0)', 'Middle (0.5)', 'Nuclear periphery (1.0)'])
            cbar.set_label('Relative Distance', rotation=270, labelpad=10)
        else:  # Absolute distance case
            cbar.set_ticklabels([f'Nuclear center (0 μm)', f'Middle ({vmax/2:.1f} μm)', f'Nuclear periphery ({vmax:.1f} μm)'])
            cbar.set_label('Distance', rotation=270, labelpad=10)
    elif color_fixed and vmin is not None:
        if len(averaged_beads) > 0:
            ax.text(max_mb * 0.95, len(chr_order) * 1.02, f'All beads at {vmin:.2f} μm', ha='right', va='bottom', fontsize=9, color='red')
    
    plt.title(title if title else f"Average Bead Radial Position Heatmap\nNuclear radius: {vmax:.1f} μm")
    plt.tight_layout()
    
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"\nFigure successfully saved to: {output_filename}")
    except Exception as e:
        print(f"\nError saving figure: {e}")
    
    plt.close(fig)

def process_multiple_samples(sample_dirs, lad_bed_file, nuclear_radius=None):
    """Process multiple sample directories and calculate absolute distances"""
    all_samples_data = {}
    lad_absolute_distances = defaultdict(list)  # Store absolute distances for LAD regions
    sample_lad_absolute_distances = defaultdict(list)  # Store LAD region distances by sample
    
    for sample_dir in tqdm(sample_dirs, desc="Processing sample directories"):
        if not os.path.exists(sample_dir):
            print(f"Warning: Sample directory does not exist: {sample_dir}")
            continue
            
        cmm_files = glob.glob(os.path.join(sample_dir, "*.cmm"))
        if not cmm_files:
            print(f"Warning: No .cmm files found in directory {sample_dir}")
            continue
            
        print(f"\nProcessing sample directory: {sample_dir}")
        print(f"Found {len(cmm_files)} .cmm files")
        
        # Process files for current sample
        averaged_beads, calculated_radius = process_multiple_files(cmm_files)
        
        # Use specified nuclear radius or calculated radius
        current_radius = nuclear_radius if nuclear_radius is not None else calculated_radius
        
        # Calculate absolute distances
        for bead in averaged_beads:
            chr_id, start, end, r = bead
            
            # Check if within LAD region
            for lad_start, lad_end in [(s, e) for c, s, e in lad_regions.keys() if c == chr_id]:
                if (start <= lad_end and end >= lad_start):  # Overlap
                    if not np.isnan(r):
                        lad_absolute_distances[(chr_id, lad_start, lad_end)].append(r)
                        sample_lad_absolute_distances[sample_dir].append(r)
        
        all_samples_data[sample_dir] = {
            'nuclear_radius': current_radius
        }
    
    # Calculate average absolute distance for each LAD region
    lad_average_distances = {}
    for lad_region, distances in lad_absolute_distances.items():
        valid_distances = [d for d in distances if not np.isnan(d)]
        if valid_distances:
            lad_average_distances[lad_region] = {
                'mean_absolute': np.mean(valid_distances),
                'std_absolute': np.std(valid_distances),
                'count': len(valid_distances)
            }
    
    return all_samples_data, lad_average_distances, sample_lad_absolute_distances

def plot_lad_boxplot(sample_lad_absolute_distances, output_dir):
    """Plot boxplot of absolute distances for LAD regions"""
    plt.figure(figsize=(12, 6))
    
    # Prepare data
    sample_names = []
    distances = []
    for sample_dir, dists in sample_lad_absolute_distances.items():
        sample_names.append(os.path.basename(sample_dir))
        distances.append(dists)
    
    # Plot boxplot
    box = plt.boxplot(distances, labels=sample_names, patch_artist=True)
    
    # Set boxplot colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(sample_names)))
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    # Add statistical information
    for i, dists in enumerate(distances):
        mean = np.mean(dists)
        std = np.std(dists)
        plt.text(i+1, plt.ylim()[1], f'n={len(dists)}\nμ={mean:.2f}±{std:.2f} μm', 
                ha='center', va='bottom', fontsize=8)
    
    plt.title('LAD Region Absolute Distance Distribution by Sample')
    plt.ylabel('Absolute Distance (μm)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save boxplot
    output_file = os.path.join(output_dir, 'lad_absolute_distance_boxplot.pdf')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def save_lad_distances_bed(lad_average_distances, sample_dirs, output_dir):
    """Save absolute distances for LAD regions to BED file"""
    # Prepare sample name list
    sample_names = [os.path.basename(d) for d in sample_dirs]
    
    # Create output file
    bed_file = os.path.join(output_dir, 'lad_distances.bed')
    with open(bed_file, 'w') as f:
        # Write header
        header = ['#chrom', 'start', 'end'] + sample_names
        f.write('\t'.join(header) + '\n')
        
        # Write data for each LAD region
        for (chr_id, start, end), stats in sorted(lad_average_distances.items()):
            # Basic position information
            row = [chr_id, str(start), str(end)]
            # Add average distance for each sample
            for sample_dir in sample_dirs:
                sample_name = os.path.basename(sample_dir)
                distances = [d for d in lad_absolute_distances.get((chr_id, start, end), []) 
                           if not np.isnan(d)]
                mean_dist = np.mean(distances) if distances else 'NA'
                row.append(f'{mean_dist:.2f}' if isinstance(mean_dist, float) else 'NA')
            f.write('\t'.join(row) + '\n')
    
    return bed_file

def process_and_plot_distances(all_samples_data, lad_regions, lad_average_distances, sample_lad_absolute_distances, sample_dirs, output_dir):
    """Process distance data and generate plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save LAD region statistics
    lad_stats_file = os.path.join(output_dir, "lad_region_statistics.txt")
    with open(lad_stats_file, 'w') as f:
        f.write("LAD Region Statistics\n")
        f.write("===================\n\n")
        for (chr_id, start, end), stats in sorted(lad_average_distances.items()):
            f.write(f"Region: {chr_id}:{start}-{end}\n")
            f.write(f"Mean absolute distance: {stats['mean_absolute']:.3f} ± {stats['std_absolute']:.3f} μm\n")
            f.write(f"Number of measurements: {stats['count']}\n\n")
    
    # Plot boxplot
    boxplot_file = plot_lad_boxplot(sample_lad_absolute_distances, output_dir)
    print(f"Absolute distance boxplot saved to: {boxplot_file}")
    
    # Save BED file
    bed_file = save_lad_distances_bed(lad_average_distances, sample_dirs, output_dir)
    print(f"LAD region distance data saved to: {bed_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python process_multiple_cmm.py <input_directories> <lad_bed_file> [nuclear_radius]")
        print("Example: python process_multiple_cmm.py ./sample1,./sample2,./sample3 ./LADs.bed")
        print("Or: python process_multiple_cmm.py ./sample1,./sample2,./sample3 ./LADs.bed 5.0")
        sys.exit(1)
    
    input_dirs = sys.argv[1].split(',')
    lad_bed_file = sys.argv[2]
    
    # Check if input directories exist
    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            print(f"Error: Input directory does not exist: {input_dir}")
            sys.exit(1)
    
    # Check if LAD file exists
    if not os.path.exists(lad_bed_file):
        print(f"Error: LAD BED file does not exist: {lad_bed_file}")
        sys.exit(1)
    
    # Check if nuclear radius parameter is provided
    nuclear_radius = None
    if len(sys.argv) == 4:
        try:
            nuclear_radius = float(sys.argv[3])
            if nuclear_radius <= 0:
                raise ValueError("Nuclear radius must be greater than 0")
            print(f"Using specified nuclear radius: {nuclear_radius:.2f} μm")
        except ValueError as e:
            print(f"Error: Invalid nuclear radius value - {e}")
            sys.exit(1)
    
    # Read LAD regions
    lad_regions = read_lad_bed(lad_bed_file)
    print(f"Read {len(lad_regions)} LAD regions from {lad_bed_file}")
    
    # Process all samples
    all_samples_data, lad_average_distances, sample_lad_absolute_distances = process_multiple_samples(input_dirs, lad_bed_file, nuclear_radius)
    
    # Create output directory
    output_dir = "absolute_distance_plots"
    process_and_plot_distances(all_samples_data, lad_regions, lad_average_distances, sample_lad_absolute_distances, input_dirs, output_dir)
    
    print(f"\nAll output files saved to directory: {output_dir}")