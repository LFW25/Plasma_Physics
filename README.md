# Plasma Simulation Project

This project models plasma physics phenomena, focusing on the two-stream instability model. The core is built using a configurable class, `PlasmaSimulation`, optimized for visualizing plasma behavior and running simulations in a GUI. 

## Features

- **Core Class: `PlasmaSimulation`**  
  This class simulates plasma interactions by calculating particle positions, velocities, electric fields, and tracking energy evolution. A GUI mode has been added for live visualization.
  
- **GUI Mode**  
  The project includes a `SimulationGUI` to support real-time plotting, phase-space analysis, and energy evolution.  
  - **Phase Space Plot**: Displays position vs. velocity.
  - **Energy Plot**: Shows kinetic, potential, and total energy over time.

## Installation

### Prerequisites

The code requires Python 3.x and several scientific libraries. It is recommended to use Anaconda or Miniconda to manage dependencies.

### Setting Up a Conda Environment

To set up with Conda:

1. Create and activate a Conda environment:
   ```bash
   conda create -n plasma_sim python=3.x
   conda activate plasma_sim
   ```

2. Install required packages:
   ```bash
   conda install numpy scipy matplotlib
   ```

### Cloning the Repository

Clone the repository to your local machine:
```bash
git clone <repository_url>
cd <repository_name>
```

## Running the Code

The main simulation code is in `twostream-optimized.py`.

### Basic Usage

1. **Edit Configuration**: Update the configuration parameters in the `config` dictionary within `twostream-optimized.py` to suit your simulation needs.
2. **Run the Simulation**: 
   ```bash
   python twostream-optimized.py
   ```

### GUI Mode

To use the GUI for live visualization, start `twostream-optimized.py`, and a GUI will open where you can interact with the simulation. 

## Configuration Parameters

The simulation allows several configurable parameters:
- `NG`, `NT`, `N`: grid points, timesteps, and particles count.
- `dens`, `L`, `V0`, `VT`, `XP1`: plasma density, grid length, stream velocities, thermal velocity, and position parameters.
- `viz`: a dictionary with visualization settings (e.g., marker size, transparency).

## Example Configuration

```python
config = {
    "NG": 64,
    "NT": 200,
    "N": 100,
    "dens": 1.0,
    "L": 10.0,
    "V0": 0.1,
    "VT": 0.05,
    "XP1": 0.5,
    "DT": 0.1,
    "viz": {
        "stream1_color": "firebrick",
        "stream2_color": "dodgerblue",
        "marker_size": 1.5,
        "alpha": 0.7,
        "phase_xlim": 10.0,
        "phase_ylim": 2.0
    }
}
```

## Classes and Methods

### PlasmaSimulation

**Initialization**:  
`PlasmaSimulation(config, gui_mode=False)`

- `config`: Dictionary of simulation parameters.
- `gui_mode`: Enables real-time GUI updating if set to `True`.

**Methods**:
- `run_simulation`: Runs the simulation with live plot updates.
- `stop_simulation`: Stops the simulation.
- `update`: Advances simulation by one timestep.
- `init_animation`: Initializes animation with an empty plot.

## License

This project is open-source and distributed under the MIT License.
