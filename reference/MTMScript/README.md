# Overview
This is the source code for RAL paper ***A Reliable Gravity Compensation Control Strategy for dVRK Robotic Arms with Nonlinear Disturbance Forces***. All the figures in the experiement mentioned in the paper can be reproduced using the MATLAB source code.

# Software Requirement

We have tested all the code in Matlab 2018a. User are recommended to reproduce the code using the following Matlab version:

- Matlab 2018a or higher 

Data have included in the source files. There are no additional data or software dependency which are required to furtherly install.


# How to run
## Reproduce the figures in the paper

- To plot the *Fig.3 Torque vs Position on Joint 4* in the paper, run in Matlab:

```
plot_cable_torques
```


- To plot the *Fig.  5.Comparison  of  one-joint  and  two-joint  data  collection  approaches*， run in Matlab 

```
plot_dataCollection_RMS.
```


- To plot the *Fig. 7.   RMS Absolute Error between the measured and predicted joint torqueof Joint 5 modeled with different polynomial function order* in the paper, run in Matlab:

```
plot_optimized_pol_order
```


- To plot the *Fig. 8.Joint trajectory for the MTM in the trajectory test* and *Fig.  9.Comparison  of  the  GCC  output  joint  torques  obtained  by  CAD,Fontanelli et al. [11] and our method with measured torques.*, run in the paper, run in Matlab:

```
plot_traj_test2
```

- To plot the *Fig. 10.    Results of the drift test obtained by CAD, Fontanelli et al. [11] andour method.*, run in the paper, run in Matlab:

```
plot_drift_test2
```


# Data collected in our experiments of our paper
They can be found in the **./data/** folder.



