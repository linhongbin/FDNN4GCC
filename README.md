# DFNN4GCC

## overview

* Experiment codes for paper, *Learning Deep Nets for Gravitational Dynamics with Disturbance through Physical Knowledge Distillation*
* A package implementing Deep Feedforward Neural Network for Gravity Compensation Control (DFNN4GCC) in Master Tool Manipulator (MTM) in daVinci Research Kit (dVRK).

## Requirement
* Ubuntu OS
* ROS
* python == 2.7
* Matlab (including robotics toolbox to run ROS API)

## Installation
1. Install [dVRK](https://github.com/jhu-cisst/cisst/wiki/Compiling-cisst-and-SAW-with-CMake#13-building-using-catkin-build-tools-for-ros).
2. Implement the [analytical solution](https://github.com/jhu-dvrk/dvrk-gravity-compensation) of GCC for dVRK (To obtain a Physical Teacher Model).
3. Install required Python packages
```bash
cd DFNN4GCC
pip install requirements.txt
```
4. Install DFNN4GCC
```
git clone https://github.com/linhongbin-ws/DFNN4GCC
```

## Run
1. open a terminal
  ```bash
  roscore
  ```
  open another terminal and launch the dVRK console to control the MTM.
  ```bash
  qlacloserelays
  rosrun dvrk_robot dvrk_console_json -j <path-to-your-MTM-json-file>
  ```
2. open Matlab. Go to the "DFNN4GCC" directory. Type in the command line
  ```
  rosinit
  addpath('<path to /dvrk-ros>')
  ```

3. modify `ARM_NAME`('MTML' or 'MTMR') and `SN`('Serial Number') in the file "run_collectData.m", for example:
  ```
  ARM_NAME = 'MTMR'
  SN = '41878'
  ```

4. run `run_collectData.m`

    In this program, we collect training, validating, testing data for a MTM. It take around **4** hours to finish the process. There are 4 subprocesses running in serial, `wizard program`, `generating pivot points`, `Collision Checking`, `data collection`, `data pre-processing`.

    * `wizard program` (required command inputs): A wizard program for setting the customized joint limits for specific dVRK system. This is important since it can identify the maximum joint ranges within a safety workspace. In anther words, it helps to improve the balance between safety and achieved performance. User need to type character to input some commands in the command dialog.

    * `generating pivot points`: Generate the pivot points representing the desired positions of a MTM for training, validating and testing data.

    * `Collision Checking` (might require to press E-stop): Run through some pivot points to check if MTM will hit environment in the future data collection. Press E-stop if the MTM hit environment.

    * `data collection`: Collecting data. It should take around 4+ hours. If you pass the `Collision Checking`, you no longer need to worry about that the MTM will hit environment during this 4 hours. You can do others work waiting the program.

    * `data pre-processing`: Pre-processing the raw data to trigonometric representation.

5. run `run_train.py`

    Modify `ARM_NAME`('MTML' or 'MTMR') and `SN`('Serial Number') in the file `run_train.py`, for example:
  ```
  ARM_NAME = 'MTMR'
  SN = '41878'
  ```

    Run the program to train DFNN for Learn-from-Sratch(LfS) and Phyiscal-Knowledge-Distllation(PKD). Type in your terminal:
  ```bash
  chmod +x run_train.py
  python run_train.py
  ```

6. copy the json file for the [analytical solution](https://github.com/jhu-dvrk/dvrk-gravity-compensation) to DFNN4GCC directory. For example:
In the terminal

    ```bash
    cp <path-to-json-file>/gc-MTMR-31519.json <path-to-DFNN4GCC>/data/MTMR_31519/real)
    ```

7. run `run_Controller.py`

     Run this script to run your GCC.
     * Modify `ARM_NAME`('MTML' or 'MTMR'), `load_PTM_param_path` in the file `run_Controller.py`. For example:

    ```python
    MTM_ARM = 'MTMR'
    load_PTM_param_path = join("data", "MTMR_31519", "real", "gc-MTMR-31519.json")
    ```

  * uncomment the `controller_type` you want to use, for example

        ```python
        # controller_type = 'LfS' # Learn-from-Sratch approach
        controller_type = 'PKD' # Physical Knowledge Distillation
        # controller_type = 'PTM' # Physical Teacher Model
        ```

  *  In the terminal, Type

        ```bash
        chmod +x run_Controller.py
        python run_Controller.py
        ```

  Type `Ctrl+C` to stop safely.
