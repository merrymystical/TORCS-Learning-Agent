# Learning-Based Autonomous Driving Agent for TORCS

A learning-based autonomous driving system developed for **TORCS (The Open Racing Car Simulator)** using **Behavior Cloning (Imitation Learning)** and **Scikit-Learn Neural Networks**. The project collects driving telemetry from human demonstrations, trains a neural network on the recorded data, and enables the vehicle to autonomously predict steering, acceleration, braking, and gear control.

---

##  Project Overview

Traditional autonomous driving systems rely heavily on handcrafted rules and heuristics. This project explores an alternative approach by allowing an artificial intelligence model to **learn driving behavior directly from human driving demonstrations**.

The workflow consists of:

1. Recording telemetry data while manually driving inside TORCS.
2. Storing the telemetry as a structured dataset.
3. Training a neural network using supervised learning.
4. Deploying the trained model to autonomously control the vehicle.

The objective is to transition from a **rule-based driver** to a **learning-based autonomous driving agent**.

---

##  Objectives

- Collect driving telemetry from TORCS.
- Build a structured driving dataset.
- Learn vehicle control using supervised learning.
- Replace manual driving rules with model predictions.
- Enable autonomous navigation across different racing environments.

---

#  Machine Learning Approach

This project implements **Behavior Cloning (Imitation Learning)**.

Instead of programming driving rules manually, the neural network learns the mapping:

```
Vehicle State
        ↓
Neural Network
        ↓
Driving Commands
```

### Learning Algorithm

- Scikit-Learn
- Multi-Layer Perceptron (MLPRegressor)

### Neural Network Configuration

| Parameter | Value |
|------------|-------|
| Model | MLPRegressor |
| Hidden Layers | (128,128) |
| Activation | ReLU |
| Optimizer | Adam |
| Learning Type | Supervised Learning |

---

#  Inputs to the Model

The neural network receives environmental observations from TORCS, including:

- 19 Track Edge Sensors
- 36 Opponent Distance Sensors
- Vehicle Speed
- Vehicle Angle
- Track Position
- Engine RPM

**Total Features:** 59

---

#  Predicted Outputs

The trained model predicts:

- Steering Angle
- Acceleration
- Brake Pressure

Gear selection is handled automatically using predefined speed thresholds.

---

#  Dataset

Driving demonstrations are recorded while manually driving inside TORCS.

Each sample contains:

- Vehicle state
- Track sensors
- Opponent sensors
- Position
- Speed
- RPM
- Steering command
- Acceleration command
- Brake command
- Gear
- Reward

Example dataset:

```
telemetry_log.csv
```

Additional driving sessions are stored as JSONL files for analysis and replay.

---

#  Project Workflow

```
Manual Driving
        │
        ▼
Telemetry Collection
        │
        ▼
CSV Dataset
        │
        ▼
Feature Extraction
        │
        ▼
MLP Neural Network Training
        │
        ▼
model.pkl
        │
        ▼
Autonomous Driving
```

---

#  Repository Structure

```
TORCS-Learning-Agent/
│
├── pyclient.py
├── driver.py
├── enhancedDriver.py
├── improvedCarPredict.py
├── model.py
├── msgParser.py
├── carState.py
├── carControl.py
│
├── telemetry_log.csv
├── telemetry_log.jsonl
├── torcs_data_*.jsonl
│
├── model.pkl
│
├── gear_command.txt
├── track.txt
│
└── README.md
```

---

#  File Descriptions

### `pyclient.py`

Main client responsible for communication between TORCS and the AI agent.

Responsibilities:

- Connects to TORCS using UDP sockets.
- Exchanges simulator messages.
- Trains the neural network if a trained model does not exist.
- Loads the trained model.
- Sends predicted control commands back to TORCS.

---

### `driver.py`

Core autonomous driving module.

Responsibilities:

- Parses incoming telemetry.
- Extracts environmental features.
- Predicts steering, acceleration, and braking using the trained neural network.
- Performs automatic gear selection.
- Logs telemetry for future training.

---

### `enhancedDriver.py`

Experimental version of the driver used for testing alternative driving strategies and control improvements.

---

### `model.py`

Defines utilities related to machine learning models, including training, prediction, and model persistence.

---

### `improvedCarPredict.py`

Prototype prediction module used for experimenting with different machine learning approaches and performance improvements.

---

### `msgParser.py`

Parses incoming TORCS messages and converts them into structured data for the driving agent.

---

### `carState.py`

Maintains the current state of the vehicle, including:

- Speed
- RPM
- Gear
- Track sensors
- Opponent sensors
- Vehicle orientation

---

### `carControl.py`

Formats and sends control commands back to TORCS, including:

- Steering
- Acceleration
- Braking
- Gear
- Clutch

---

### `telemetry_log.csv`

Primary dataset generated from manual driving sessions. Used for training the neural network.

---

### `telemetry_log.jsonl`

JSON Lines representation of the telemetry dataset for debugging and data inspection.

---

### `torcs_data_*.jsonl`

Telemetry recordings from multiple driving sessions. Useful for replay, analysis, and dataset expansion.

---

### `gear_command.txt`

Stores gear-related configuration used during manual driving experiments.

---

### `track.txt`

Stores the default track configuration used during testing.

---

#  Installation

Clone the repository

Install dependencies

---

# Running the Project

### Step 1

Launch TORCS.

### Step 2

Select the desired track and vehicle.

### Step 3

Run the client:

```bash
python pyclient.py
```

If `model.pkl` is not present:

```
telemetry_log.csv
        ↓
Training
        ↓
model.pkl
```

Otherwise, the trained model is loaded directly and used for autonomous driving.

---

#  Technologies Used

- Python
- TORCS
- NumPy
- Pandas
- Scikit-Learn
- Joblib
- UDP Socket Programming

---

#  Future Improvements

- Deep Reinforcement Learning (PPO, SAC, DDPG)
- Sequence Models (LSTM/GRU)
- Computer Vision Integration
- Camera-Based Driving
- Lane Detection
- Obstacle Avoidance
- Domain Randomization
- Multi-Track Generalization

