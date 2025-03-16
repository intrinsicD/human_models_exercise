#!/usr/bin/env python3
import os
import pickle
import argparse
import numpy as np

def LoadSmplxParams(file_path):
    """
    Loads SMPLX parameters from a pickle file.
    Expects the file to contain a dictionary with keys such as:
    "betas", "global_orient", and "body_pose".
    """
    with open(file_path, 'rb') as f:
        params = pickle.load(f)
    for key in params:
        params[key] = np.array(params[key], dtype=float)
    return params

def LoadSmplxParams(pkl_file):
    """
    Loads SMPLX parameters from a pickle file.
    Expects a dictionary with keys such as 'betas', 'global_orient', and 'body_pose'.
    """
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    # Convert all entries to numpy arrays of type float.
    for key in data:
        data[key] = np.array(data[key], dtype=float)
    return data

def AverageSmplxParams(file_list):
    """
        Exercise: Compute the average SMPLX parameters from a list of pickle files.

        Each pickle file contains a dictionary where the keys are parameter names (such as "betas",
        "global_orient", "body_pose", etc.) and the values are numerical arrays (typically NumPy arrays).

        Your task is to implement this function following these steps:

        1. Initialize an empty dictionary called 'sum_params' to store the sum of parameters across files.
           Also initialize a counter 'count' to keep track of how many files have been processed.

        2. Loop over each file in the provided 'file_list':
           a. Load the parameters from the file using the provided helper function 'LoadSmplxParams(file_path)'.
           b. If this is the first file (i.e. 'sum_params' is empty), initialize 'sum_params' as a copy of the
              loaded parameters (make sure to copy the arrays so that you don't modify the original data).
           c. For every subsequent file, iterate over all keys in the dictionary and add the current file's parameter
              values to the corresponding key in 'sum_params'.
           d. Increment the counter 'count' by one.

        3. After processing all files, if 'count' is still zero (i.e. no files were provided), raise a ValueError.

        4. Create a new dictionary 'avg_params' by dividing each value in 'sum_params' by the count of files.

        5. Return the 'avg_params' dictionary.

        Example:
            file_list = ['params1.pkl', 'params2.pkl', 'params3.pkl']
            avg_params = AverageSmplxParams(file_list)
        """
    sum_params = {}
    count = 0
    avg_params = {}
    for file_path in file_list:
        # TODO: Implement ...
        pass

    return avg_params

def SaveAvgParams(avg_params, output_path):
    """
    Saves the averaged parameters to a pickle file.
    """
    with open(output_path, 'wb') as f:
        pickle.dump(avg_params, f)
    print(f"Saved average parameters to '{output_path}'")

def GenerateMeshFromParams(model, avg_params):
    """
    Creates a SMPLX model from the specified model folder and gender,
    and generates a mesh using the averaged parameters.
    """

    model_output = model(**avg_params)
    vertices = model_output.vertices.detach().cpu().numpy().squeeze()

    return vertices, model.faces

def WriteObj(obj_filename, vertices, faces):
    """
    Writes the mesh (vertices and faces) into an OBJ file.
    """
    with open(obj_filename, 'w') as f:
        for v in vertices:
            f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        for face in faces:
            # OBJ faces are 1-indexed.
            f.write("f {} {} {}\n".format(face[0] + 1, face[1] + 1, face[2] + 1))
    print(f"OBJ file saved as '{obj_filename}'")

def Main():
    parser = argparse.ArgumentParser(
        description="Compute average SMPLX parameters from fitted models and generate a mesh OBJ."
    )
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing the fitted SMPLX .pkl files (searched recursively)")
    parser.add_argument("--model_folder", type=str, required=True,
                        help="Path to the SMPLX model folder (e.g., models_smplx_v1_1/models)")
    parser.add_argument("--gender", type=str, required=True,
                        choices=["male", "female", "neutral"],
                        help="Gender for the SMPLX model")
    parser.add_argument("--output_avg", type=str, default="avg_params.pkl",
                        help="Output pickle file for averaged parameters")
    parser.add_argument("--output_obj", type=str, default="avg_mesh.obj",
                        help="Output OBJ file name")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run the model on ('cpu' or 'cuda')")

    args = parser.parse_args()

    # Collect all .pkl files from the results directory recursively.
    file_list = []
    for root, dirs, files in os.walk(args.results_dir):
        for file in files:
            if file.endswith(".pkl"):
                file_list.append(os.path.join(root, file))
    if not file_list:
        print(f"No .pkl files found in {args.results_dir}")
        return

    print(f"Found {len(file_list)} parameter files. Computing average parameters...")
    avg_params = AverageSmplxParams(file_list)

    # Save averaged parameters.
    SaveAvgParams(avg_params, args.output_avg)

if __name__ == "__main__":
    Main()
