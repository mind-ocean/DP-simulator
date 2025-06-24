# DP-simulator

This script allows for DP prognosis with DNV-ST-0111 level 3 

The script is initiated from main_input.py, the ship input data are determined there at the beginning of the script. The description of each variable is given at the top of the script and in the dissertation that is also provided there. The script is only a part of the PHD that covers a subject of prognosis with all DNV levels esspecially. Program requite an input.csv file that can be only generated with level 1 with the other provided script - ship-dynamic-positioning-capability-prognosis. Program uses Ansys-Aqwa software that generates ANALYSIS.LIS file that is futher used in the simulator for wave forces calculation. Another file to provide is wind.csv that contains coefficient of wind forces calculated according to IMCA 140.

The rest of the .py files contain classes and functions that are called in man_input.py and main_sim_ver3.py

The results are found in the out/{SHIP_NAME} folder. Some examples are already given there.

Python version supporting the script is 3.8.12

The list of libraries and their versions is given in file dp_libraries.txt

The script may be developed further or used as it is. The functionality is correct and results validated. The performance in sense of calculation time significant.

WARNING!
Examplary MSV ship given, is calculated (simulated) in model scale of 1:36. This was done due to validation purposes with the phycal model of the ship in scale. To introduce new design one must also modify the controller and observer data, ship and thrusters models ect.
