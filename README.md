# Anomaly Detection Project

This project implements various anomaly detection algorithms for assessing the potential fraudulence of financial transactions. The implemented algorithms include Z-Score, Isolation Forest, DBSCAN, and LOF (Local Outlier Factor).

## Deployment

The Streamlit app is deployed and accessible [here](https://anomalydetectionkayser.streamlit.app/).

## Overview

- [Introduction](#introduction)
- [Implemented Algorithms](#implemented-algorithms)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Anomaly detection is a crucial task in identifying potentially fraudulent activities within financial transactions. This project provides a streamlined implementation of different algorithms to identify anomalies in transaction data.

## Implemented Algorithms

### 1. Z-Score Method

The Z-Score method assesses the potential fraudulence of financial transactions based on specific transaction features. It calculates Z-scores for each feature, categorizing them into customer profile, customer engagement, credit card usage, and transaction history. The result indicates whether the transaction is normal or potentially fraudulent, specifying the category if fraudulent.

### 2. Isolation Forest

Isolation Forest is an algorithm that isolates anomalies by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature. The code predicts anomalies for user inputs in categories such as customer profile, customer engagement, credit card usage, and transaction history.

### 3. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

DBSCAN is a density-based clustering algorithm that groups together points that are close to each other based on a distance measure and a minimum number of points. This project uses DBSCAN to detect outliers in specific columns of the dataset.

### 4. LOF (Local Outlier Factor)

LOF is a density-based algorithm that measures the local density deviation of a data point with respect to its neighbors. This project utilizes LOF for anomaly detection in financial transactions, determining if the last row of user input is considered an outlier based on the LOF score.

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/anomaly-detection-project.git
   cd anomaly-detection-project

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the Streamlit app:
   ```bash
   streamlit run main.py

### How It Works

The project uses Streamlit to create a user-friendly interface for inputting values for different transaction features. Users can input values, and the app predicts whether the transaction is normal or potentially fraudulent based on the selected algorithm.


### Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit a pull request.

### License

This project is licensed under the [MIT License](LICENSE).
