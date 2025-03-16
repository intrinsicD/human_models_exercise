# Exercise - Human Models

In this exercise, you will make smpl-x work and fit smpl-x models to images.

## Task

Make sure `libprotobuf-dev protobuf-compiler libopencv-dev libboost-all-dev libhdf5-dev libatlas-base-dev` are installed

1. copy all files from `human_models_data.zip` into `human_models_exercise`
2. extract the files from `models_smplx_v1_1.zip` into `human_models_exercise`
2. In the `human_models_exercise` folder run the command `git submodule update --init --recursive` to fetch the complete content of the submodules
   It should download the smplify-x, openpose, human_body_prior, torch-mesh-isect and cuda_samples submodules and their dependencies

## Compile openpose
1. In `human_models_exercise/openpose/models` copy the `.caffeemodel` files from `openpose_models`  into `pose/body_25/` `face/` and `hand` folder
2. In `human_models_exercise/openpose` create a folder called `build` with `mkdir build && cd build`
3. In `human_models_exercise/openpose/build` configure openpose using `cmake .. -DUSE_CUDNN=OFF` (if you did not copy the modelfiles by hand this will download the BODY_25 models and will sadly take around 20 minutes or more!!!)
4. In `human_models_exercise/openpose/build` build openpose with `make -j$(nproc)`
5. In `human_models_exercise/openpose` generate the keypoints of the test image with `./build/examples/openpose/openpose.bin --image_dir ../data/images --write_json ../data/keypoints/ --display 0 --render_pose 0 --face 0 --hand 0`
6. In `human_models_exercise/data/keypoints` check if the file `rgb_7_keypoints.json` was generated

## Create a python virtual environment
1. In `human_models_exercise` create a python environment with `conda create -n human_models_exercise_env python=3.11 pytorch torchvision torchaudio cudatoolkit=11.7 -c pytorch -c nvidia`
2. Activate the environment with `conda activate human_models_exercise_env`
3. Make sure that `echo $CUDA_SAMPLES_INC` outputs the path to `cuda-samples/Common`. If it is not, run `export CUDA_SAMPLES_INC=~/absolute_path_to/human_models_exercise/cuda-samples/Common` by replacing `absolute_path_to` with the path where you have `human_models_exercise` located
3. Run `pip install -r requirements.txt` to install the required dependencies
4. In `miniconda3/envs/human_models_exercise_env/lib/python3.11/site-packages/torchgeometry/core/conversions.py` you need to replace the following lines in the function `rotation_matrix_to_quaternion`. Replace all occurrances of `(1 - mask_d0_d1)` with `(~mask_d0_d1)` and 
`(1 - mask_d2)` with `(~mask_d2)` and `(1 - mask_d0_nd1)` with `(~mask_d0_nd1)`
5. In `miniconda3/envs/human_models_exercise_env/lib/python3.11/site-packages/pyrender/mesh.py`  replace all occurrances of `np.infty` with `np.inf`
5. Make sure that `echo $DISPLAY` outputs something like `:0`. if not run `export DISPLAY=:0`
6. In `human_models_exercise` now run the following command: `python smplify-x/smplifyx/main.py --config smplify-x/cfg_files/fit_smplx.yaml --data_folder data/ --output_folder output/ --visualize="True" --model_folder models_smplx_v1_1/models --vposer_ckpt vposer_v1_0/ --part_segm_fn smplx_parts_segm.pkl` to check the whole pipeline.

## Exercise

In `demo.py`:

1. Implement the function `AverageSmplxParams`.

2. Run `python demo.py --results_dir output/results/ --model_folder models_smplx_v1_1//models --gender female --device cuda` to compute the averaging of all the smplx parameters to get a better fit for one frame

3. Run `python smplify-x/smplifyx/render_pkl.py --config smplify-x/cfg_files/fit_smplx.yaml --model_folder models_smplx_v1_1/models --vposer_ckpt vposer_v1_0/ --part_segm_fn smplx_parts_segm.pkl --pkl avg_params.pkl` to render a the averaged pkl file

<br/>
<center><h3>Good Luck!</h3></center>
