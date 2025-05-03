# SSVEP Experiment Simulation

A visualization and simulation tool for SSVEP (Steady-State Visual Evoked Potential) EEG experiments. This application allows you to visualize EEG data alongside a simulation of the flickering LED stimuli that were presented during the original experiments.

## Requirements

- Python 3.12 or higher
- Pygame
- NumPy
- mat73

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/SSVEP_hackathon.git
cd SSVEP_hackathon
```

2. Set up a virtual environment:

```bash
# Using uv (recommended)
uv init --no-workspace --python 3.12
source .venv/bin/activate

# Or using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
# Using uv
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

## Usage

Run the simulation:

```bash
python simulation.py
```

The application will load available datasets from the `data/` directory. If no datasets are found, it will use default paths.

## Controls

- **Space**: Pause/resume the simulation
- **R**: Restart the simulation from the beginning
- **Left/Right Arrows**: Navigate between trials
- **Up/Down Arrows**: Increase/decrease playback speed
- **Number Keys (0-9)**: Manually select trigger channel
- **ESC**: Exit the simulation

## Data Visualization

The simulation displays:

- **Left side**: LED simulation showing 4 flickering stimulus boxes (9Hz, 10Hz, 12Hz, 15Hz)
- **Right side**: Real-time EEG channel data display
- **Bottom**: Timeline showing current position and trial markers

## Dataset Structure

The simulation expects MATLAB .mat files in the `data/` directory containing:

- Variable `y` with EEG data in the format: [Sample Time, 8 EEG channels, Trigger, LDA]
- The trigger channel (typically channel 9) indicates when stimuli are presented

The LED sequence for each trial is loaded from `data/classInfo_4_5.m` if available, or a default sequence is used.

## Troubleshooting

If the trigger channel is not correctly detected:

- Use number keys (0-9) to manually select the correct trigger channel
- Check that your data files contain properly formatted EEG data with trigger markers
