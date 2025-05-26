# 🛫 Flight Price Prediction – Project Summary

## 🔍 Overview
This project predicts flight ticket prices using machine learning models trained on airline booking data. The system extracts and processes features like airline, source, destination, number of stops, and travel date to estimate ticket fares. The model is deployed via a Streamlit web app that allows users to input trip details and get a fare prediction.

---

## 📦 Key Features
- **Interactive Streamlit Web App**: For real-time price predictions based on user inputs.
- **Machine Learning Models**:
  - Linear Regression
  - Random Forest Regressor (with GridSearchCV tuning)
  - Support Vector Regressor
  - Polynomial Regression (Degree 2 & 3)
- **Data Visualization**: Comparative bar charts for airline prices and number of stops.

---

## 🧹 Data Preprocessing
- **Missing Values**: Filled using the most frequent values in each column.
- **Feature Engineering**:
  - Extracted `day`, `month`, and `year` from `Date_of_Journey`.
  - Mapped categorical features (Airline, Source, Destination, etc.) to numerical values.
  - Converted `Duration` to total hours in float format.
- **Dropped Columns**: `Route`, `Dep_Time`, `Arrival_Time` for simplicity.

---

## 🧠 Model Training
- **Train-Test Split**: 80% training and 20% testing data.
- **Evaluation Metrics**:
  - `.score()` method for R² accuracy
  - Visual inspection using line and scatter plots
- **Best Performing Model**: Random Forest Regressor (tuned)

---

## 🧪 Hyperparameter Tuning
Used `GridSearchCV` to tune the Random Forest Regressor. Best parameters:

```python
{
  'n_estimators': 200,
  'min_samples_split': 5,
  'min_samples_leaf': 1,
  'max_features': 'log2',
  'max_depth': 20
}
```

## 📊 Visualizations
- **Bar Plots**:
  - Airline vs Price
  - Number of Stops vs Price
- **Correlation Heatmap**: Shows relationships between numerical features
- **Prediction Accuracy Plots**:
  - Scatter and line plots comparing predicted and actual values

---

---

## 🖥️ Streamlit App
The Streamlit web application is designed to be user-friendly and responsive. It includes:

- **Dropdowns** for Airline, Source, Destination
- **Date Picker** for Travel Date
- **Select Box** for Ticket Type (Economy / Business)
- **Number Inputs** for Adult and Child count
- **Slider** for Maximum Number of Stops
- **Text Input** for Additional Info (optional)
- **"Predict Price" Button** to trigger prediction
- **Output Display** showing:
  - Estimated Flight Price
  - A bar chart comparing prices across airlines
### 🔢 Inputs:
- Airline
- Source / Destination
- Travel Date
- Ticket Type: Economy / Business (25% price increase)
- Adult and Child Count
- Max Number of Stops
- Additional Info (optional)

### 📤 Output:
- Predicted Price
- Airline-wise price comparison bar chart

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Denistanb/flight-price-prediction.git
cd flight-price-prediction
```
 
### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn streamlit jupyter
```

### 3. Run the Notebook or Python Script
To train models:  
```bash
jupyter notebook "Price_Detection.ipynb"
```
To launch the Streamlit app:  
```bash
streamlit run app.py
```
