# Reimplementing PPO and Assessing its Performance in Stock Trading Strategies

This project implements a Proximal Policy Optimization (PPO) algorithm for training agents in various environments. It includes comparisons between our PPO implementation and Stable Baselines' implementation on the Pendulum and custom StockEnv environments.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project implements a Proximal Policy Optimization (PPO) algorithm for training agents in different environments. The primary focus is on comparing the performance of our PPO implementation with Stable Baselines' implementation on the Pendulum and custom StockEnv environments.

## Features

- Implements Proximal Policy Optimization (PPO) algorithm
- Compares performance with Stable Baselines' implementation
- Evaluates on Pendulum and custom StockEnv environments

## Installation

To install and set up the project, follow these steps:

1. Clone the repository:
```sh
$ git clone https://github.com/your_username/your_project.git
```
2. Install dependencies: 
```sh
$ pip install -r requirements.txt
```
3. Create the dataset:
```sh
$ python utils/create_dataset.py
```
to generate the dataset required for the custom stock environment.

## Usage

To use the project, follow these steps:

1. Train the agents on the StockEnv environment using [`train.py`](train.py) with appropriate parameters.
2. Evaluate performance on the StockEnv environment using [`evaluate.py`](evaluate.py) to compare performance metrics.
3. Perform experiments on the Pendulum environment using [`pendulum_experiments.py`](pendulum_experiments.py) to train and evaluate agents.
4. Modify model and environment configurations in the [`configs`](configs) folder as needed for custom experiments and configurations.

## Contributing

Contributions are welcome! If you find any bugs, have feature requests, or want to contribute code, please follow these guidelines:

1. Check for existing issues or create a new one.
2. Fork the repository and create a new branch for your feature.
3. Make your changes and test thoroughly.
4. Submit a pull request with a clear description of your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
