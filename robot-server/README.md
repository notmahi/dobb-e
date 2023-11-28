<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!-- PROJECT LOGO -->
<!-- <br /> -->
<div align="center">
  <h1 align="center">Hello Stretch Server</h3>
  <p align="center">
    Code to start the camera stream publisher and robot controller. The code is useful for record3d based camera streaming and controller but can be adapted for other use cases.
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## Instruction for Installation and Running

First clone the repository to your hello robot. Using `requirements.txt`, you can install the required packages.
**Important:** Make sure the requirements are installed in the root python environment (the same environment where `stretch_body` package is installed) on your stretch robot. Otherwise, this code will not work.

To run the server, follow the following steps:
* Make sure your robot is [joint-calibrated](https://github.com/hello-robot/stretch_body/blob/master/tools/bin/stretch_robot_home.py) by running 
  ```sh
  stretch_robot_home.py
  ```
* Once calibrated, run ```roscore``` in an independent terminal
* Then, in a new terminal, cd to the hello-stretch-server directory and run 
  ```sh
  python3 start_server.py
  ```
