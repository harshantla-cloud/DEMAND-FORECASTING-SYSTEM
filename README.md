# 📊 Demand Forecasting System

> A machine learning-based application that predicts product demand using pricing, promotion, inventory, competitor pricing, and product category data.

## Overview

Demand forecasting helps businesses estimate future product demand and support better inventory and planning decisions.

This project uses a machine learning model to predict product demand based on selected product and market-related features. The trained model is integrated into an interactive Streamlit application where users can enter feature values and receive an instant demand forecast.

## Key Features

- Predicts product demand using a trained machine learning model
- Interactive Streamlit-based user interface
- Accepts real-time user inputs for prediction
- Supports categorical product categories using label encoding
- Uses pricing, discount, inventory, promotion, and competitor pricing features
- Displays forecasted demand in units
- Includes exploratory data analysis and machine learning notebooks

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Pickle

## Project Structure

```text
DEMAND-FORECASTING-SYSTEM/
│
├── assets/
│   └── background.png
│
├── data/
│   ├── demand_forecasting.csv
│   └── preprocessed_demand_forcasting_data.csv
│
├── models/
│   ├── xgboost_demand_model.pkl
│   └── label_encoder.pkl
│
├── notebooks/
│   ├── analysis.ipynb
│   └── machine_learning.ipynb
│
├── app.py
├── requirements.txt
└── README.md
```

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/harshantla-cloud/DEMAND-FORECASTING-SYSTEM.git
```

### 2. Navigate to the Project Directory

```bash
cd DEMAND-FORECASTING-SYSTEM
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

## Usage

1. Enter the product price
2. Enter the discount percentage
3. Enter the inventory level
4. Select promotion status
5. Enter competitor pricing
6. Select the product category
7. Click **Predict Demand**
8. View the predicted demand

## Model

The project uses an **XGBoost Regressor (XGBRegressor)** for demand prediction.

### Input Features

- Price
- Discount
- Inventory Level
- Promotion
- Competitor Pricing
- Category

The categorical product category is processed using a saved label encoder before prediction.

## Notebooks

The project includes notebooks for:

- Exploratory Data Analysis
- Data Preprocessing
- Feature Engineering
- Machine Learning Model Training
- Model Evaluation

## Demo

Interactive Streamlit application for real-time demand prediction.

## Future Improvements

- Add batch prediction using CSV uploads
- Include additional time-series features
- Add advanced visualizations
- Implement model monitoring
- Add more production-ready features

## Author

**Harsh**

GitHub: https://github.com/harshantla-cloud

---

⭐ If you found this project useful, consider giving it a star!
