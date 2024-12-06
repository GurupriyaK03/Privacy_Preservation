# Privacy_Preservation
PRIVACY PRESERVATION

# Federated Learning with TensorFlow and Keras  

This repository contains implementations of federated learning algorithms using TensorFlow and Keras on the **Credit Card Fraud Detection Dataset**. The project demonstrates various techniques for distributed learning, including:

- Federated Averaging (FedAvg)
- Federated Proximal (FedProx)
- Federated SGD (FedSGD)

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Setup](#setup)
5. [Usage](#usage)
6. [Federated Learning Algorithms](#federated-learning-algorithms)
7. [Results](#results)
8. [Contributions](#contributions)
9. [License](#license)

## Introduction
Federated learning enables decentralized training of machine learning models across multiple clients without sharing raw data. This project focuses on binary classification of fraudulent transactions using a federated approach.

## Features
- Implementation of three federated learning techniques:
  - **FedAvg**: Aggregates model weights from multiple clients.
  - **FedProx**: Introduces a proximal term to handle heterogeneous data.
  - **FedSGD**: Aggregates gradients instead of weights.
- Utilizes **TensorFlow** for neural network creation and training.
- Demonstrates model evaluation across local and global datasets.
- Handles real-world imbalanced datasets with stratified splitting.

## Dataset
The dataset used is the **Credit Card Fraud Detection Dataset**, which contains anonymized features representing transaction data.

### Source:
You can download the dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).

### Preprocessing:
- Standardization of numerical features (excluding the `Time` column).
- Handling missing values in the target column (`Class`).

## Setup
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- TensorFlow 2.x
- Pandas
- NumPy
- Scikit-learn

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/federated-learning.git
   cd federated-learning
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset and place it in the root folder as `creditcard.csv`.

## Usage
1. Run the preprocessing script:
   ```bash
   python preprocess.py
   ```
2. Train a specific federated model:
   - For **FedAvg**:
     ```bash
     python train_fedavg.py
     ```
   - For **FedProx**:
     ```bash
     python train_fedprox.py
     ```
   - For **FedSGD**:
     ```bash
     python train_fedsgd.py
     ```
3. View evaluation results in the console.

## Federated Learning Algorithms
### FedAvg
Aggregates model weights from local clients to update the global model.  
- **Pros**: Simple and effective.  
- **Cons**: Assumes homogeneous data distribution.

### FedProx
Adds a proximal term to the loss function to address data heterogeneity.  
- **Pros**: Handles client variability better than FedAvg.  

### FedSGD
Aggregates gradients instead of model weights for global updates.  
- **Pros**: Reduces communication cost.  

## Results
The results include:
- Local model accuracies for each client.
- Global model accuracy after each federated round.

| Algorithm | Final Global Accuracy |
|-----------|------------------------|
| FedAvg    | 0.95                  |
| FedProx   | 0.96                  |
| FedSGD    | 0.94                  |

## Contributions
Feel free to open an issue or submit a pull request to contribute to this project.  

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

