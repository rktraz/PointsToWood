import datetime
start = datetime.datetime.now()
import resource
import os
import os.path as OP
import argparse
from src.preprocessing import *
from src.predicter import SemanticSegmentation
from tqdm import tqdm
import torch
import shutil
import sys
import numpy as np
import re
import json
from src.io import load_file, save_file

# Add a debug function
def debug_point_cloud(pc_file, pc_data):
    """Print debug information about a point cloud file"""
    print(f"\n--- Debug info for {pc_file} ---")
    print(f"Point cloud shape: {pc_data.shape}")
    print(f"Point cloud columns: {pc_data.columns.tolist()}")
    print(f"First few points:\n{pc_data.head()}")
    print(f"Min/max values for x,y,z:")
    for col in ['x', 'y', 'z']:
        if col in pc_data.columns:
            print(f"  {col}: {pc_data[col].min()} to {pc_data[col].max()}")
    print("----------------------------\n")

def set_num_threads(num_threads):
    torch.set_num_threads(num_threads)
    os.environ['OMP_NUM_THREADS'] = str(num_threads)

def create_timestamped_output_dir(base_dir="inference_runs"):
    """Create a timestamped output directory for inference runs"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, timestamp

def save_run_info(output_dir, args, input_files, timestamp, runtime_seconds=None, classification_results=None):
    """Save run information to info.txt file"""
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        else:
            return obj
    
    run_info = {
        "timestamp": timestamp,
        "runtime_seconds": runtime_seconds,
        "input_files": input_files,
        "parameters": {
            "batch_size": args.batch_size,
            "num_procs": args.num_procs,
            "resolution": args.resolution,
            "grid_size": args.grid_size,
            "min_pts": args.min_pts,
            "max_pts": args.max_pts,
            "model": args.model,
            "is_wood": args.is_wood,
            "any_wood": args.any_wood,
            "output_fmt": args.output_fmt,
            "verbose": args.verbose
        },
        "classification_results": classification_results
    }
    
    # Convert numpy types in classification_results
    if classification_results:
        run_info["classification_results"] = convert_numpy_types(classification_results)
    
    info_file = os.path.join(output_dir, "info.txt")
    with open(info_file, 'w') as f:
        f.write("PointsToWood Inference Run Information\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        if runtime_seconds:
            f.write(f"Runtime: {runtime_seconds:.2f} seconds\n")
        f.write(f"Input files: {input_files}\n\n")
        f.write("Parameters:\n")
        f.write("-" * 20 + "\n")
        for param, value in run_info["parameters"].items():
            f.write(f"{param}: {value}\n")
        
        if classification_results:
            f.write("\nClassification Results:\n")
            f.write("-" * 20 + "\n")
            for result in classification_results:
                f.write(f"File: {result['file']}\n")
                f.write(f"  Total points: {result['total_points']}\n")
                f.write(f"  Wood points: {result['wood_points']} ({result['wood_percent']:.2f}%)\n")
                f.write(f"  Non-wood points: {result['non_wood_points']} ({result['non_wood_percent']:.2f}%)\n\n")
        
        f.write("\n" + "=" * 50 + "\n")
    
    # Also save as JSON for programmatic access
    json_file = os.path.join(output_dir, "run_info.json")
    with open(json_file, 'w') as f:
        json.dump(run_info, f, indent=2)
    
    print(f"\nRun information saved to: {info_file}")
    print(f"JSON run info saved to: {json_file}")

# Add a function to calculate class distribution
def calculate_class_distribution(output_file):
    """
    Calculate and print the distribution of wood vs non-wood points in the output file.
    """
    try:
        # Load the classified point cloud
        classified_pc, _ = load_file(filename=output_file, additional_headers=True, verbose=False)
        
        # Check if the file has the expected classification columns
        if 'pwood' in classified_pc.columns:
            # Count wood points (pwood == 1) and non-wood points (pwood == 0)
            wood_points = (classified_pc['pwood'] == 1).sum()
            total_points = len(classified_pc)
            non_wood_points = total_points - wood_points
            
            # Calculate percentages
            wood_percent = (wood_points / total_points) * 100
            non_wood_percent = (non_wood_points / total_points) * 100
            
            print("\n--- Classification Results ---")
            print(f"Total points: {total_points}")
            print(f"Wood points: {wood_points} ({wood_percent:.2f}%)")
            print(f"Non-wood points: {non_wood_points} ({non_wood_percent:.2f}%)")
            print("-----------------------------")
        else:
            print("\nWarning: 'pwood' column not found in the output file. Cannot calculate class distribution.")
    except Exception as e:
        print(f"\nError calculating class distribution: {str(e)}")

'''
Minor functions-------------------------------------------------------------------------------------------------------------
'''

def get_path(location_in_pointstowood: str = "") -> str:
    current_wdir = os.getcwd()
    match = re.search(r'PointsToWood.*?pointstowood', current_wdir, re.IGNORECASE)
    if not match:
        raise ValueError('"PointsToWood/pointstowood" not found in the current working directory path')
    last_index = match.end()
    output_path = current_wdir[:last_index]
    if location_in_pointstowood:
        output_path = os.path.join(output_path, location_in_pointstowood)
    return output_path.replace("\\", "/")

def preprocess_point_cloud_data(df):
    df.columns = df.columns.str.lower()
    columns_to_drop = ['label', 'pwood', 'pleaf']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    df = df.rename(columns=lambda x: x.replace('scalar_', '') if 'scalar_' in x else x)
    df = df.rename(columns={'refl': 'reflectance', 'intensity': 'reflectance'})
    headers = [header for header in df.columns[3:] if header not in columns_to_drop]
    if 'reflectance' not in df.columns:
        df['reflectance'] = np.zeros(len(df))
        print('No reflectance detected, column added with zeros.')
    else:
        print('Reflectance detected')
    cols = list(df.columns)
    if 'reflectance' in cols:
        cols.insert(3, cols.pop(cols.index('reflectance')))
        df = df[cols]
    return df, headers, 'reflectance' in df.columns

'''
-------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------
'''

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--point-cloud', '-p', default=[], nargs='+', type=str, help='list of point cloud files')    
    parser.add_argument('--odir', type=str, default='.', help='output directory')
    parser.add_argument('--batch_size', default=8, type=int, help="If you get CUDA errors, try lowering this.")
    parser.add_argument('--num_procs', default=-1, type=int, help="Number of CPU cores you want to use. If you run out of RAM, lower this.")
    parser.add_argument('--resolution', type=float, default=None, help='Resolution to which point cloud is downsampled [m]')
    parser.add_argument('--grid_size', type=float, nargs='+', default=[2.0, 4.0], help='Grid sizes for voxelization')
    parser.add_argument('--min_pts', type=int, default=512, help='Minimum number of points in voxel')
    parser.add_argument('--max_pts', type=int, default=9999999, help='Maximum number of points in voxel')
    parser.add_argument('--model', type=str, default='model.pth', help='path to candidate model')
    parser.add_argument('--is-wood', default=0.5, type=float, help='a probability above which points within KNN are classified as wood')
    parser.add_argument('--any-wood', default=1, type=float, help='a probability above which ANY point within KNN is classified as wood')
    parser.add_argument('--output_fmt', default='ply', help="file type of output")
    parser.add_argument('--verbose', action='store_true', help="print stuff")

    args = parser.parse_args()

    # Configure the number of threads based on args.num_procs
    if args.num_procs == -1:
        num_threads = os.cpu_count()
    else:
        num_threads = args.num_procs

    set_num_threads(num_threads)

    if args.verbose:
        print('\n---- parameters used ----')
        for k, v in args.__dict__.items():
            if k == 'pc': v = '{} points'.format(len(v))
            if k == 'global_shift': v = v.values
            print('{:<35}{}'.format(k, v)) 

    args.wdir = get_path()
    args.mode = 'predict' if 'predict' in sys.argv[0] else 'train'
    args.reflectance = False

    '''
    Sanity check---------------------------------------------------------------------------------------------------------
    '''
    if args.point_cloud == '':
        raise Exception('no input specified, please specify --point-cloud')
    
    for point_cloud_file in args.point_cloud:
        if not os.path.isfile(point_cloud_file):
            raise FileNotFoundError(f'Point cloud file not found: {point_cloud_file}')
    
    '''
    Create timestamped output directory
    '''
    output_base_dir, timestamp = create_timestamped_output_dir()
    print(f"\nOutput directory created: {output_base_dir}")
    
    # Save initial run info
    save_run_info(output_base_dir, args, args.point_cloud, timestamp)
    
    # Initialize list to store classification results
    classification_results = []
    
    '''
    If voxel file on disc, delete it.
    '''    
    
    path = OP.dirname(args.point_cloud[0])
    args.vxfile = OP.join(path, "voxels")

    if os.path.exists(args.vxfile): shutil.rmtree(args.vxfile)

    for point_cloud_file in args.point_cloud:

        '''
        Handle input and output file paths-----------------------------------------------------------------------------------
        '''
        
        # Create output filename in timestamped directory
        input_filename = OP.splitext(OP.basename(point_cloud_file))[0]
        output_filename = input_filename + "_wood.ply"
        args.odir = OP.join(output_base_dir, output_filename)
        args.h5 = OP.join(output_base_dir, output_filename.replace('_wood.ply', '_features.h5'))
        
        '''
        Preprocess data into voxels------------------------------------------------------------------------------------------
        '''

        if args.verbose: print('\n----- Preprocessing started -----')

        os.makedirs(args.vxfile, exist_ok=True)
        args.pc, args.headers = load_file(filename=point_cloud_file, additional_headers=True, verbose=False)
        
        # Add debug info
        debug_point_cloud(point_cloud_file, args.pc)
        
        args.pc, args.headers, args.reflectance = preprocess_point_cloud_data(args.pc)
        
        # Add debug info after preprocessing
        print("\n--- After preprocessing ---")
        print(f"Point cloud shape: {args.pc.shape}")
        
        print(f'Voxelising to {args.grid_size} grid sizes')
        preprocess(args)
        
        # Add debug info after voxelization
        print(f"\n--- After voxelization ---")
        print(f"Number of voxel files: {len(os.listdir(args.vxfile))}")
        
        if args.verbose:
            print(f'peak memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}')
            print(f'runtime: {(datetime.datetime.now() - start).seconds}')
        
        '''
        Run semantic training------------------------------------------------------------------------------------------------
        '''
        if args.verbose: print('\n----- Semantic segmenation started -----')
        
        # Try with a lower min_pts if the original fails
        try:
            SemanticSegmentation(args)
        except ValueError as e:
            if "need at least one array to concatenate" in str(e):
                print("\n!!! ERROR: No valid voxels found for segmentation !!!")
                print("Trying with a lower min_pts value...")
                
                # Save original min_pts
                original_min_pts = args.min_pts
                
                # Try with a lower min_pts
                args.min_pts = max(64, args.min_pts // 4)
                print(f"Retrying with min_pts={args.min_pts}")
                
                # Recreate voxels with new min_pts
                if os.path.exists(args.vxfile):
                    shutil.rmtree(args.vxfile)
                os.makedirs(args.vxfile, exist_ok=True)
                preprocess(args)
                
                # Try segmentation again
                SemanticSegmentation(args)
                
                # Restore original min_pts
                args.min_pts = original_min_pts
            else:
                # If it's a different error, re-raise it
                raise
                
        torch.cuda.empty_cache()

        if os.path.exists(args.vxfile):
            shutil.rmtree(args.vxfile)

        # Calculate and print class distribution for the output file
        if os.path.exists(args.odir):
            # Get classification results
            try:
                classified_pc, _ = load_file(filename=args.odir, additional_headers=True, verbose=False)
                if 'pwood' in classified_pc.columns:
                    wood_points = (classified_pc['pwood'] == 1).sum()
                    total_points = len(classified_pc)
                    non_wood_points = total_points - wood_points
                    wood_percent = (wood_points / total_points) * 100
                    non_wood_percent = (non_wood_points / total_points) * 100
                    
                    # Store results
                    classification_results.append({
                        'file': os.path.basename(args.odir),
                        'total_points': total_points,
                        'wood_points': wood_points,
                        'wood_percent': wood_percent,
                        'non_wood_points': non_wood_points,
                        'non_wood_percent': non_wood_percent
                    })
                    
                    calculate_class_distribution(args.odir)
                else:
                    print(f"\nWarning: 'pwood' column not found in {args.odir}")
            except Exception as e:
                print(f"\nError processing classification results: {str(e)}")
        else:
            print(f"\nOutput file not found: {args.odir}")

        if args.verbose:
            print(f'peak memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}')
            print(f'runtime: {(datetime.datetime.now() - start).seconds}')

    # Calculate final runtime and save run info after all files are processed
    runtime_seconds = (datetime.datetime.now() - start).total_seconds()
    save_run_info(output_base_dir, args, args.point_cloud, timestamp, runtime_seconds, classification_results)
    
    print(f"\nFinal runtime: {runtime_seconds:.2f} seconds")
    print(f"All results saved to: {output_base_dir}")
