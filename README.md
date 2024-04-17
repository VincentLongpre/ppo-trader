# Reimplementing PPO and Assessing its Performance in Stock Trading Strategies

By [Alexandre Diz Ganito](https://github.com/Dizga), [Vincent Longpré](https://github.com/VincentLongpre) and [Igman Talbi](https://github.com/Iggy1055).

Final project, COMP579 : Reinforcement Learning - Winter 2024, Prof. Doina Precup. McGill University

## Table of Contents

- [Introduction](#introduction)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to reimplement the Proximal Policy Optimization (PPO) algorithm from scratch using PyTorch. Our goal is to achieve comparable results to Stable Baselines' implementation across multiple environments. We conduct comparisons between our PPO implementation and Stable Baselines' version on the Pendulum environment and a custom StockEnv environment containing the 30 constituent stocks of the Dow Jones index.

## Models

In our project, we developed two distinct classes to facilitate experimentation and comparison:

1. **PPO:** This class represents our final implementation of the Proximal Policy Optimization (PPO) algorithm. It incorporates strategies and configurations that were identified through iterative testing as contributing to improved performance.

2. **PPO_test:** This class serves as a sandbox environment for testing and experimenting with various strategies inspired by Stable Baselines' implementation of PPO. We used this class to explore different configurations, activation functions, policy distribution variances, and other parameters to understand their impact on performance.




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
2. Evaluate performance on the StockEnv environment using [`test.py`](evaluate.py) to compare performance metrics.
3. Perform experiments on the Pendulum environment using [`pendulum_experiments.py`](pendulum_experiments.py) to train and agents and visualize learning curves.
4. Modify model and environment configurations in the [`configs`](configs) folder as needed for custom experiments and configurations.

## Contributing

Contributions are welcome! If you find any bugs, have feature requests, or want to contribute code, please follow these guidelines:

1. Check for existing issues or create a new one.
2. Fork the repository and create a new branch for your feature.
3. Make your changes and test thoroughly.
4. Submit a pull request with a clear description of your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
