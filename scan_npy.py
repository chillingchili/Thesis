import os
import numpy as np

def scan_pose_dataset(source_dir, expected_joints=33, expected_dims=2):
    """
    Scans a directory for .npy files and returns a list of corrupted 
    or misaligned tensors for manual review.
    """
    corrupted_files = []
    scanned_count = 0
    
    for filename in os.listdir(source_dir):
        if not filename.endswith('.npy'):
            continue
            
        filepath = os.path.join(source_dir, filename)
        scanned_count += 1
        is_valid = True
        
        try:
            data = np.load(filepath)
            
            # Check 1: Dimensionality
            if len(data.shape) != 3:
                is_valid = False
            elif data.shape[1] != expected_joints or data.shape[2] != expected_dims:
                is_valid = False
                
            # Check 2: Missing values (NaNs)
            elif np.isnan(data).any():
                is_valid = False
                
            # Check 3: Total tracking failure (absolute zeros)
            elif np.all(data == 0):
                is_valid = False
                
        except Exception:
            # Catches natively unreadable files
            is_valid = False
            
        if not is_valid:
            corrupted_files.append(filepath)
            shape_info = data.shape if 'data' in locals() else 'Unreadable'
            print(f"Flagged: {filename} | Shape: {shape_info}")
            
    print(f"\nScan complete. Scanned {scanned_count} files. Found {len(corrupted_files)} corrupted files.")
    return corrupted_files

def delete_flagged_files(file_list):
    """
    Deletes the exact list of files passed to it.
    """
    if not file_list:
        print("No files to delete.")
        return
        
    # Optional safety prompt
    confirm = input(f"Are you sure you want to permanently delete these {len(file_list)} files? (y/n): ")
    if confirm.lower() != 'y':
        print("Deletion cancelled.")
        return

    deleted_count = 0
    for filepath in file_list:
        try:
            os.remove(filepath)
            deleted_count += 1
            print(f"Deleted: {os.path.basename(filepath)}")
        except Exception as e:
            print(f"Error deleting {os.path.basename(filepath)}: {e}")
            
    print(f"\nCleanup complete. Deleted {deleted_count} files.")

# --- Workflow Execution ---

# Step 1: Run the scan and store the output in a variable
# flagged_list = scan_pose_dataset('./training_data/drive_serves')

# Step 2: Manually check the printed 'flagged_list' in your console.
# If you agree with the flagged files, pass the list to the deletion function:
# delete_flagged_files(flagged_list)