# ArUco_Pose_Estimation
This repository contains the bare minimum code for simple pose estimation using ArUco markers in Python

<div align="center">
<img src = 'assets/aruco-track.gif ' width = 700>
</div>

## Dependencies
- OpenCV
- Numpy

## Installation

1. Download the repository:
    ```bash
    git clone https://github.com/Jshulgach/ArUco_Pose_Estimation.git
    cd ArUco_Pose_Estimation
    ```
2. Install dependencies according to your OS:
    ### Windows
    It's best to install dependencies in a virtual environment. Using either Anaconda or Python venv, prepare your environment:
    ```bash
    python -m venv aruco python=3.10
    call aruco\Scripts\activate
    pip install -r requirements.txt
    ```
    ### Linux
    Install using the `apt install` command:
    ```bash
    sudo apt install python3-opencv python3-yaml
    ```

## Usage

### Checkerboard
Make sure to have a checkerboard printed if you want to have accurate pose calibration. You can find a multitude of patterns from [Mark Hedley Jones](https://markhedleyjones.com/projects/calibration-checkerboard-collection). Once you print out the one you like, make sure to update the `calibration_settings.yaml` file with the correct dimensions of the squares on the checkerboard.

### Running the scripts
The files are named in the order of operations:

- `1_save_frames.py` : Used to save frames from the camera feed.
- `2_calibrate.py`   : Performs the calibration routine with a single camera, computes distortion matrix.
- `3_main.py`        : Detects ArUco markers in the camera feed.

You can find more details on parameters for each script using `python my-script-to-run.py --help`. Ideally all of these should be run without much to change
   
1. **Save Calibration Frames**  

    Run `1_save_frames.py` to initialize the camera feed. Press the space bar when prompted to start collecting images of your checkerboard. Make sure to move the checkerboard around to get different poses and orientations.
    ```bash
    python 1_save_frames.py
    ```
   
2. **Calibration**  
    Run `2_calibrate.py` to read the checkerboard images in your directory and generate a `calibration_matrix.npy` and `distortion_coefficients.npy` file. 
    ```bash
    python 2_calibrate.py  
    ```
    * Note: If you're connected to another device via ssh (like a raspberry pi) and need to transfer the `.npy` files, you can copy them to your local WSL environment with the command:
	    ```bash
	    scp myusername@192.168.1.164:/home/myusername/github/ArUco_Pose_Estimation/distortion_coefficients.npy /home/myusername
		```
        For copying to your host windows maching:
	    ```bash
	    scp myusername@192.168.1.164:/home/myusername/github/ArUco_Pose_Estimation/distortion_coefficients.npy /mnt/c/Users/WindowsUserName
		```    	
   
3. **Pose Estimation**  
    Run `3_main.py` to begin running the pose estimation for each ArUCo marker detected. This is done in real-time for each frame obtained from the webcam feed.  
    ```bash
    python 3_main.py  
    ```
   

   
Feel free to reach out to me in case of any issues.  
If you find this repo useful in any way please do star ⭐️ it so that others can reap it's benefits as well.

## Acknowledgements
This repository is inspired by the work of [GSNCodes](https://github.com/GSNCodes/ArUCo-Markers-Pose-Estimation-Generation-Python)

## References
1. https://docs.opencv.org/4.x/d9/d6d/tutorial_table_of_content_aruco.html
2. https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
