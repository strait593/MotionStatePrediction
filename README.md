# Motion State Prediction: Human Activity Recognition (HAR)

## Project Overview
This project focuses on classifying human motion states using tri-axial accelerometer data. By leveraging signal processing techniques and Random Forest classification, the model identifies four distinct physical activities with high precision: **Idle, Walking, Running, and Climbing Stairs**[cite: 1, 2].

The core of this project is the transformation of raw time-series sensor data into a high-dimensional feature space suitable for machine learning.

## Technical Stack
*   **Language:** Python 3.x
*   **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn[cite: 1]
*   **Algorithm:** Random Forest Classifier (100 Estimators)[cite: 1]
*   **Data Source:** Tri-axial accelerometer readings[cite: 1]

## 📊 Feature Engineering
To move from raw sensor noise to predictive power, the following statistical features were engineered for each axis (X, Y, Z)[cite: 1, 2]:

| Feature Category | Description |
| :--- | :--- |
| **Central Tendency** | Mean values across time-windows to capture baseline acceleration[cite: 1, 2]. |
| **Dispersion** | Standard Deviation and Range to identify the intensity of movement[cite: 1, 2]. |
| **Power Metrics** | **RMS (Root Mean Square)** to quantify the total power/energy of the signal[cite: 1, 2]. |
| **Aggregate Metrics** | **SMA (Signal Magnitude Area)** used as a proxy for physical activity intensity[cite: 1, 2]. |

## Model Performance
The project utilizes a **Random Forest Classifier** known for its robustness against over-fitting in high-dimensional datasets[cite: 1]. 
*   **Training/Test Split:** 80/20[cite: 1]
*   **Data Cleaning:** Implemented custom logic to resolve "ambiguous Series" errors and clean metadata from activity labels[cite: 2].
*   **Key Finding:** Signal dispersion (Std Dev) and SMA were the most influential features in distinguishing high-intensity (Running) vs. low-intensity (Idle) states[cite: 1, 2].

## Project Structure
```text
├── data/
│   └── features.csv       # Processed dataset containing 13 engineered features[cite: 2]
├── notebooks/
│   └── main_analysis.ipynb # End-to-end ML pipeline: Cleanup, EDA, Training[cite: 1]
└── README.md

## Installation & Usage

1. Clone the repository
Bash
git clone [https://github.com/strait593/MotionStatePrediction.git](https://github.com/strait593/MotionStatePrediction.git)

2.Install dependencies
Bash
pip install scikit-learn matplotlib pandas numpy

3. Run the analysis by tapping "Run all" in Jupyter Notebook
