# Lab 5 Fall 2025 - Neural Controller Configuration

This directory contains configuration files for the Pupper v3 neural controller and a deployment script that pushes changes and rebuilds the ROS2 workspace.

## Files

- **`config.yaml`** - Neural controller configuration file with settings for all controller modes (normal, three-legged, parkour, test)
- **`launch.py`** - ROS2 launch file that starts the neural controller with all necessary nodes
- **`estop_controller.cpp`** - Emergency stop controller C++ source code for switching between controllers
- **`parkour_policy.json`** - Neural network policy file for parkour mode (deployed to neural_controller/launch/)
- **`rebuild_neural_controller.py`** - Deployment and rebuild script (copies files + runs build.sh + sets up wandb)
- **`download_latest_policy.py`** - Downloads a specific policy from Weights & Biases (auto-detects user entity)
- **`deploy.py`** - Interactive script to download a policy (optional) and launch the robot

## Usage

### Quick Start: Launch the Robot

To download a policy (optional) and launch the neural controller:

```bash
python3 deploy.py
```

This interactive script will:
1. Prompt you for a run number to download a specific policy from wandb (or press Enter to skip)
2. Download the policy if a run number is provided (automatically uses your logged-in wandb entity)
3. Launch the neural controller with `ros2 launch neural_controller launch.py`

### Deploy and Rebuild

To deploy configuration files and rebuild the ROS2 workspace:

```bash
# Preview what will be done (dry run - no files modified, no build)
python3 rebuild_neural_controller.py --dry-run

# Deploy files and rebuild the workspace
python3 rebuild_neural_controller.py

# Deploy only without rebuilding
python3 rebuild_neural_controller.py --no-build
```

The script will:
1. **Deploy files**: Create backups of existing files (with `.backup` extension) and copy:
   - `config.yaml` → `/home/pi/pupperv3-monorepo/ros2_ws/src/neural_controller/launch/`
   - `launch.py` → `/home/pi/pupperv3-monorepo/ros2_ws/src/neural_controller/launch/`
   - `estop_controller.cpp` → `/home/pi/pupperv3-monorepo/ros2_ws/src/joy_utils/src/`
   - `parkour_policy.json` → `/home/pi/pupperv3-monorepo/ros2_ws/src/neural_controller/launch/`
2. **Rebuild workspace**: Run `build.sh` in `/home/pi/pupperv3-monorepo/ros2_ws/`
3. **Setup wandb**: Install (if needed) and prompt for Weights & Biases authentication for experiment tracking
4. Display a summary of the operations

### Editing Configuration

1. Edit any configuration files in this directory:
   - `config.yaml` - Controller parameters and settings
   - `launch.py` - ROS2 launch configuration
   - `estop_controller.cpp` - Emergency stop controller source
   - `parkour_policy.json` - Parkour neural network policy
2. Run `python3 rebuild_neural_controller.py` to deploy and rebuild in one step
3. The workspace will be automatically rebuilt and wandb will be configured

## Configuration Overview

### Controller Modes

The configuration supports four neural controller modes:

1. **neural_controller** - Default mode using `policy_latest.json`
2. **neural_controller_three_legged** - Three-legged locomotion mode
3. **neural_controller_parkour** - Parkour mode using `parkour_policy.json`
4. **neural_controller_test** - Test mode using `test_policy.json`

All controllers start inactive and must be activated using ROS2 controller manager commands.

### Launch Arguments

- `sim:=True/False` - Run in Mujoco simulator (True) or on real robot (False, default)
- `teleop:=True/False` - Enable teleoperation (True, default)

Example:
```bash
ros2 launch neural_controller launch.py sim:=True teleop:=True
```

## Typical Workflow

### First Time Setup
1. Edit configuration files as needed
2. Deploy and rebuild: `python3 rebuild_neural_controller.py`
3. Authenticate with wandb when prompted
4. Launch the robot: `python3 deploy.py`

### Daily Usage
1. Download latest policy and launch: `python3 deploy.py`
   - Enter a run number to test a specific policy
   - Press Enter to use the current policy

### After Configuration Changes
1. Make changes to config files
2. Redeploy: `python3 rebuild_neural_controller.py`
3. Launch: `python3 deploy.py`

## Directory Structure

```
/home/pi/lab_5_fall_2025/           # Source files (this directory)
├── config.yaml                      # Controller configuration
├── launch.py                        # Launch file
├── estop_controller.cpp             # Emergency stop controller source
├── parkour_policy.json              # Parkour neural network policy
├── rebuild_neural_controller.py     # Deployment and rebuild script
├── download_latest_policy.py        # Download policy from wandb
├── deploy.py                        # Launch robot with optional policy download
└── README.md                        # This file

/home/pi/pupperv3-monorepo/ros2_ws/src/neural_controller/launch/  # Destination
├── config.yaml                      # Deployed configuration
├── launch.py                        # Deployed launch file
└── parkour_policy.json              # Deployed parkour policy

/home/pi/pupperv3-monorepo/ros2_ws/src/joy_utils/src/  # Destination
└── estop_controller.cpp             # Deployed emergency stop controller

/home/pi/pupperv3-monorepo/ros2_ws/  # Build location
└── build.sh                         # Build script (automatically executed)
```

## Notes

- Backup files are automatically created with `.backup` extension
- The rebuild script checks for file existence before copying
- The ROS2 workspace is automatically rebuilt after deployment
- You can add more file mappings to `rebuild_neural_controller.py` if needed
- Use `--no-build` flag if you only want to deploy without rebuilding
- Weights & Biases (wandb) authentication is prompted after successful rebuild for experiment tracking
  - If wandb is not installed, the script will automatically install it using `pip3 install wandb`
  - You'll be prompted to visit https://wandb.ai/authorize to get your API key
  - You can always run `wandb login` manually later if needed
- The `download_latest_policy.py` script automatically detects your wandb entity (username) from your logged-in session
  - No need to manually configure entity names for each student
  - You can override with `--entity <name>` if needed

