# Robotic Labelling for Computer Vision Datasets
This repository contains our application to generate labelled datasets, intended for 6D pose estimation. While we try to create as large datasets as possible, the intended purpose is to create validition/testing datasets and NOT training datasets. Sampling data from these datasets is considered as defeating the purpose, since the data in these datasets is correlated. Scene share the same poses of labelled objects. Multiple Scenes are still captured in the same general environment and may probably share background images and distractors.

## Requirements
- Ubuntu 22.04
- Recommended: `conda install -c conda-forge openexr-python` 
- Python 3.10
- `pip install -r requirements.txt`
- start with `bash run_app.sh` or `streamlit run main.py`
- Install the ZED SDK according to their install instructions and the Python API (Keep in mind the USB type C socket of the camera is not reversible)

## Semantics and general info
- *Objects* are things, with annotations and therefore 3D scanned or modelled digital twins.
- *Distractors* are things without annotations, e.g. random things, placed in the scene as clutter
- The *background* is the horizontally mounted monitor.
- A *scene* is one static configuraton of labelled objects. Lighting conditions, backgrounds and distracting objects may (and should) change. As long as the labelled object(s) do not move, every datapoint is considered part of the same *scene*.
- A *datapoint* is a single capture of a scene. This may contain multiple images of multiple cameras and multiple different labels, masks, depths, etc. Lighting, the objects, the distractors, the background and the cameras must not change in a single datapoint.
- Depth cameras, such as the Intel Realsense can be configured very differently. Usually there is a trade-off between accuracy and speed. High-quality in this project refers to the best accuracy and should be used for all registration steps. Low-quality settings, that may be used in application scenerios may be used for data acquisition.

## Workflows
1. General Setup: Needs to be done (ideally) once, for each camera
    1. Camera Intrinsics (and Stereo) Calibration
    2. Handeye Calibration (Robot Flange -> Camera)
    3. Workspace/Plane Registration
2. Scene Setup: Object Pose Labelling: register the 6D pose of all *objects* in the scene
    1. Remove all distractors, only the plain object
    2. initial guess
        - from PVN3D
        - from placing in known pose (using the monitor for example)
    3. Acquire multiple datapoints
        - from different point of views
        - with plain colored backgrounds
        - with highest-quality depth
    4. For each datapoint, generate the pixel perfect mask using color thresholds
    5. Optimize the pose of the object, using ICP with the multiple frames of masked depths and the initial guess
3. Data acquisition:
    ```
    FOR i times:
        Without moving the objects (!), place randomly distractors in the Scene
        FOR j times:
            Move the robot to a new point of view
            Render the masks of the non-occluded objects
            Calculate the visible mask of the objects:
                1. Take a high-quality depth image
                2. Calculate the ground truth depth image of the non-occluded object
                3. Take the difference between HQ captured and GT calculated depth
                4. `Visible Mask := Non-Occluded mask && (difference<threshold)`
            FOR k times:
                Set a new background
                Change lighting conditions
                Take RGB, stereo, depth (low-quality) images
                Write annotations and images to disk
    ```