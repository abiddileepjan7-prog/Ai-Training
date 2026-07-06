# SCADA Pipeline K-Means Clustering

## Overview

This project demonstrates the application of **Unsupervised Machine Learning** on industrial SCADA (Supervisory Control and Data Acquisition) pipeline data. The objective is to identify hidden operational patterns by grouping similar pipeline operating conditions using the **K-Means Clustering** algorithm.

Instead of relying on predefined labels, the model discovers natural groupings within the data based on multiple sensor readings collected from pipeline operations.

---

## Project Objectives

- Perform Exploratory Data Analysis (EDA)
- Preprocess industrial SCADA sensor data
- Standardize numerical features
- Determine the optimal number of clusters using the Elbow Method
- Apply K-Means Clustering
- Visualize clusters using Principal Component Analysis (PCA)
- Evaluate clustering performance using the Silhouette Score
- Interpret operational characteristics of each cluster

---

## Dataset

The dataset contains simulated SCADA pipeline sensor readings representing various operational conditions.

### Features

- Timestamp
- Segment ID
- Pressure
- Flow Rate
- Temperature
- Valve Status
- Pump State
- Pump Speed
- Compressor State
- Energy Consumption
- Alarm Triggered
- Event Type *(excluded during training)*
- Target *(excluded during training)*

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## Machine Learning Workflow

1. Data Loading
2. Data Cleaning
3. Feature Selection
4. Feature Scaling
5. Exploratory Data Analysis
6. Elbow Method
7. K-Means Clustering
8. PCA Visualization
9. Cluster Evaluation
10. Cluster Interpretation

---

## Algorithms Used

### K-Means Clustering

Groups pipeline operating conditions into clusters based on feature similarity.

### Principal Component Analysis (PCA)

Reduces feature dimensions for visualization while preserving most of the variance.

---

## Evaluation Metric

- Silhouette Score

The Silhouette Score measures how well-separated the clusters are.

---

## Project Structure

```
SCADA-Pipeline-KMeans-Clustering/
│
├── scada_pipeline.csv
├── clustering.py
├── README.md
├── requirements.txt
└── images/
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/SCADA-Pipeline-KMeans-Clustering.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python clustering.py
```

---

## Results

The project successfully groups similar pipeline operating conditions into distinct clusters, enabling the identification of different operational states and supporting anomaly detection for industrial monitoring systems.

---

## Future Improvements

- DBSCAN Clustering
- Hierarchical Clustering
- Isolation Forest for anomaly detection
- Interactive dashboards using Plotly
- Real-time streaming analysis
- Hyperparameter optimization

---

## Author

Abid Dillep Jan

Machine Learning Intern
