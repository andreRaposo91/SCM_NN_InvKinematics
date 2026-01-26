import re
import matplotlib.pyplot as plt
from pathlib import Path

# Image paths
image_paths = [
    ('FNN3', r"results\final_datasets3\models_2024-03-26T112415\imgs\mlp_1_75pct_3L.png",),
    ('FNN6', r"results\final_datasets3\models_2024-04-09T130207\imgs\mlp_1_50pct_2L.png",),
    ('RNN', r"results\final_datasets_rnn\models_2024-04-03T155223\imgs\rnn_100pct_2L.png",),
    ('FNN3-CC', r"results\final_datasets_comp\models_2024-06-08T124704\imgs\mlp_1_75pct_4L.png",),
    ('FNN6-CC', r"results\final_datasets_comp\models_2024-06-06T190740\imgs\mlp_1_100pct_4L.png",),
    ('RNN-CC', r"results\final_datasets_rnn_comp\models_2024-06-10T152024\imgs\rnn_25pct_2L.png",),
]

def extract_info_from_path(img_path):
    """Extract date, model type, percentage, and layers from image path"""
    # Extract date from path (format: YYYY-MM-DDTHHMISS)
    date_match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{6})', img_path)
    date = date_match.group(1) if date_match else None
    print(date)
    
    # Extract filename
    filename = Path(img_path).stem
    
    # Extract percentage
    pct_match = re.search(r'(\d+)pct', filename)
    pct = float(pct_match.group(1)) if pct_match else None
    
    # Extract number of layers
    layers_match = re.search(r'(\d+)L', filename)
    num_layers = int(layers_match.group(1)) if layers_match else None
    
    # Extract folder name to find results file
    folder = Path(img_path).parent.parent.parent
    
    return {
        'date': date,
        'pct': pct,
        'num_layers': num_layers,
        'folder': folder,
        'path': img_path
    }

def find_results_file(folder, date):
    """Find the results file matching the date"""
    folder_path = Path(folder)
    if not folder_path.exists():
        return None
    
    # Look for results_*.txt files with matching date
    for results_file in folder_path.glob('results_*.txt'):
        if date in results_file.name:
            return results_file
    return None

def count_layers(arch_str):
    """Count number of layers in architecture string like '[85, 85]'"""
    # Count comma-separated values
    return len([x.strip() for x in arch_str.strip('[]').split(',')])

def parse_results_file(results_file, pct, num_layers):
    """Parse results file and find matching architectures"""
    matching_lines = []
    
    with open(results_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        # Skip header and empty lines
        if not line.strip() or 'Architechture' in line or 'Train Set' in line or 'NN Params' in line:
            continue
        
        parts = line.split(';')
        if len(parts) < 2:
            continue
        
        arch = parts[0].strip()
        pct_train = parts[1].strip()
        
        # Check if percentage matches
        try:
            if abs(float(pct_train) - pct) > 0.01:
                continue
        except ValueError:
            continue
        
        # Extract architecture pattern (e.g., [85, 85])
        arch_pattern_match = re.search(r'\[\(?[\d,\s]+.*\]', arch)
        if not arch_pattern_match:
            continue
        
        arch_pattern = arch_pattern_match.group(0)
        
        # Check if number of layers matches
        if count_layers(arch_pattern) != num_layers:
            continue
        
        matching_lines.append({
            'architecture': map(int, arch_pattern[1:-1].replace('(', '').replace(')', '').split(',')),
            'full_line': line.strip()
        })
    
    return matching_lines

# Process each image
for arch, img_path in image_paths:
    info = extract_info_from_path(img_path)
    assert Path(img_path).is_file()
    
    print(f"\nProcessing: {img_path} [{info['folder']}]")
    print(f"  Date: {info['date']}, Model: {arch}, "
          f"Pct: {info['pct']}%, Layers: {info['num_layers']}")
    
    # Find results file
    results_file = find_results_file(info['folder'], info['date'])
    
    if not results_file:
        print(f"  WARNING: No results file found for date {info['date']} in {Path(info['folder']).absolute()}")
        continue
    
    print(f"  Found results file: {results_file}")
    
    # Parse and find matching architectures
    matches = parse_results_file(results_file, info['pct'], info['num_layers'])
    
    if not matches:
        print(f"  WARNING: No matching architectures found")
        continue
    
    print(f"  Found {len(matches)} matching architecture(s)")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Dummy plot (you can edit this later)
    for m in matches:
        ax.plot([0, 1], [0, 1], label=f"{arch} - {info['pct']}% - {info['num_layers']}L - {next(m['architecture'])}N")
    
    # Create legend with architectures
    # legend_labels = [f"{m['architecture']}" for m in matches]
    
    # Add legend
    # ax.legend(loc='best', title='Architectures')
    ax.legend()

    plt.tight_layout()
    plt.show()
    

print("\n=== Processing complete ===")