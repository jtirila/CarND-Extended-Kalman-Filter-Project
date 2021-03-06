# Extended Kalman Filter Project 

**J-M Tirilä**

This is my repository for my 1st project of the Udacity Self Driving Car Nanodegree Program, Term 2.

Other participants and the instructors will know what the project is about so I am not repeating the instructions here. 
Just keeping the very broad overall picture of what the code is about.  
Also, for ease of experiments, the basic build and run instructions can be found below.  

## Brief Description of the Project

This project is about implementing a sensor fusion Kalman filter to estimate the state of a moving object of interest with noisy lidar and radar measurements. Passing the project requires obtaining RMSE values that are lower that the tolerance outlined in the project rubric. 

## Some Notes Concerning My Solution

I have not changed the template code layout that much. Just implemented the TODOs. 

I did make some changes, however, as summarized below.

 * I removed some member variable matrices from the `FusionEKF` class. In my opinion, these are conceptually 
   rather members of the `KalmanFilter` class and the ones used to initialise the `KalmanFilter` (`kf`)instance can just 
   as well be defined as local variables in FusionEKF that are discarded immediately after `kf` initialization.  
 * I refactored the common part of `KalmanFilter::Update()` and `KalmanFilter::UpdateEKF()` functions into a separate
   `KalmanFilter::DoUpdate()` function to avoid repetitive code. 
 * Instead of using an instance of Tools class for the helper method invocations (coordinate conversions, 
    RMSE and Jacobian calculation), I refactored
   the class into a _namespace_ as the methods we wish to use are essentially just static methods
   with no need for internal state, and hence no need for a class instance. 

## Technical instructions

This project involves the Term 2 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases)

This repository includes two files that can be used to set up and install [uWebSocketIO](https://github.com/uWebSockets/uWebSockets) for either Linux or Mac systems. For windows you can use either Docker, VMware, or even [Windows 10 Bash on Ubuntu](https://www.howtogeek.com/249966/how-to-install-and-use-the-linux-bash-shell-on-windows-10/) to install uWebSocketIO. Please see [this concept in the classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77) for the required version and installation scripts.
Once the install for uWebSocketIO is complete, the main program can be built and run by doing the following from the project top directory.

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./ExtendedKF


### Other Dependencies

* cmake >= 3.5
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

### Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make` 
   * On windows, you may need to run: `cmake .. -G "Unix Makefiles" && make`
4. Run it: `./ExtendedKF `
