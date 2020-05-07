# Epidemics_Simulation
This is the final project of 15-618 Parallel Computer Architecture and Programming. We build a simulation system to simulate the spread of epidemics.

## Files Architecture
We have two folders here, one is "simulation", achieving person-level CUDA+OpenMP version, the other "simulation_node", achieving node-level CUDA version.

## How to generate data
All our data is stored in "data" folder. You can generate graphs and rats data by running write_data.py with instruction\
```shell script
python write_data.py
```
You can also modify the size and the number of hubs by modifying write_data.py

## How to run code
In simulation folder
```shell
make
# Run person-level simulation
./crun-omp-gpu -g [graph_file] -r [rat_file] -u b -n [iteration_number] -t [thread_number] -I -q -F [output_file]
# Visualize the simulation
./crun-omp-gpu -g [graph_file] -r [rat_file] -u b -n [iteration_number] -t [thread_number] -I -q -F [output_file] | ./grun.py -d -v h
```
In simulation_node folder
```shell
make
# Run node-level simulation
./crun-gpu -g [graph_file] -r [rat_file] -u b -n [iteration_number] -I -q
```
