# Flight Price Prediction

**Predicts Indian domestic flight fares from trip details, served as a Streamlit app on top of a random forest model.**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

## Overview

Airline ticket prices move around a lot. They depend on the carrier, the route, the number of stops, the time of year and how far ahead you book. This project takes a historical flight-fare dataset (the MachineHack-style Indian domestic flights data), runs it through the usual machine learning pipeline, and ends with a regression model that estimates a price for a given trip.

A Streamlit app wraps the model so you don't need the notebook to use it. You pick an airline, source, destination, booking date, ticket type, passenger counts and stop preference, press **Predict**, and get an estimated fare with a few pricing rules applied on top. The notebook (`Price_Detection.ipynb`) covers the EDA, feature engineering, a comparison of four model families and hyperparameter tuning. A random forest scored about **R² 0.85** on the test split and is the model saved for the app.

I built this as a learning and portfolio project (it maps to an iNeuron.ai ML internship). The repo has the trained model, the app, a folder of design documents and a demo video.

## Key Features

- **End-to-end regression pipeline**: raw CSV data through cleaning, feature engineering, model comparison, tuning and a saved model file.
- **Four model families compared**: Linear Regression, Random Forest Regressor, Support Vector Regressor and Polynomial Regression (degree 2 and 3), all on the same 80/20 split.
- **Hyperparameter tuning**: `GridSearchCV` over the Random Forest. It scored slightly lower than the plain forest, so the saved model is the untuned one.
- **Streamlit app**: dropdowns for airline, origin and destination, a date picker, a ticket-type select, adult and child counts and a max-stops input, styled with custom CSS (teal `#00ADB5` theme).
- **Pricing rules on top of the model**: the model output is adjusted with simple booking rules (see below).
- **Per-passenger fare**: the predicted fare is multiplied by the number of adults and children.
- **Business-class surcharge**: 25% is added when "Business" is selected instead of "Economy".
- **Last-minute surge**: the fare is doubled if the travel date is less than a day away.
- **Airline price comparison chart**: a bar chart of price by airline, drawn from the training data.
- **Design documentation**: a `docs/` folder with architecture, HLD, LLD and workflow documents, a project report deck and a demo video.
- **Codespaces setup**: a `.devcontainer/devcontainer.json` that installs the requirements and starts the Streamlit app on port 8501.

## How It Works

### Data and preprocessing

The data lives in `train/Data_Train.csv` and `test/Test_set.csv` (both also provided as `.xlsx`), plus a `Sample_submission` file. Each row is a flight with an airline, source, destination, route, departure and arrival times, duration, number of stops, additional info and the target `Price`.

Preprocessing in the notebook:

- **Missing values** are filled with the most frequent value in each column.
- **Date features**: the date of journey is split into **Day**, **Month** and **Year** columns so the model can pick up seasonal effects.
- **Categorical encoding**: Airline, Source, Destination and Additional_Info are mapped to integers. The same maps are hard-coded in the app so user input is encoded the way the training data was.
- **Duration** is converted into a single number of hours.
- **Dropped columns**: Route, Dep_Time and Arrival_Time are removed to keep the feature set small.

### Model training and selection

Everything is trained on an 80/20 train-test split and scored mainly with R². The notebook compares:

| Model | Test R² (approx.) |
|---|---|
| Linear Regression | 0.38 |
| Polynomial Regression (deg 2 / 3) | 0.57 / 0.36 |
| Support Vector Regressor | not recorded in the notebook output |
| **Random Forest Regressor (`random_state=0`)** | **0.85** |
| Random Forest after `GridSearchCV` | 0.84 |

The forest was clearly ahead, so it got a `GridSearchCV` pass. The best parameters found:

```
{
  'n_estimators': 200,
  'min_samples_split': 5,
  'min_samples_leaf': 1,
  'max_features': 'log2',
  'max_depth': 20
}
```

The tuned model scored slightly below the plain forest, so the file saved to `model/best_model.pkl` (~15 MB) is the plain forest (`ran_model`). That's the file the app loads at startup, so serving predictions needs no retraining.

### The Streamlit app

`app.py` is the deployment layer. On load it unpickles `best_model.pkl`, applies a custom CSS block for the teal theme and renders a two-column input form. When you click **Predict** it:

1. Maps your selections to the same integer codes used in training (the airline, source, destination and additional-info dictionaries live in the app).
2. Builds a single-row `DataFrame` with the encoded inputs plus the split-out Day / Month / Year.
3. Calls `best_model.predict(...)` to get a base fare.
4. Applies the pricing rules: add 25% for business class, multiply by the passenger count, and double the total if the trip is less than a day out.
5. Shows the final fare (with `st.balloons()` on success) and draws the airline-vs-price bar chart underneath.

Known issue: the notebook model is trained on nine features including `Duration`, but `app.py` sends eight columns and no `Duration`. The predict call may fail with a feature mismatch. I haven't run the app to confirm.

### Pricing logic, concretely

The model predicts a single base fare and the app turns that into a total:

```
base = model.predict(inputs)
if Business:  base *= 1.25            # business-class surcharge
fare = base * (adults + children)     # per-passenger
if days_until_travel < 1: fare *= 2   # last-minute surge
```

These multipliers are rough rules I wrote into the app. They were not learned from data.

## Results / Highlights

- **Random Forest Regressor, R² ≈ 0.85** on the test split, the best of the four model families tried.
- It beat the linear baseline (R² ≈ 0.38) and polynomial regression (≈ 0.57 at degree 2). The grid-searched forest (≈ 0.84) did not improve on it.
- The model is saved as a ~15 MB pickle, so the app loads it instantly.
- Trained on about 10,700 rows of Indian domestic flights across 12 airline labels.
- Delivered as a working app rather than only a notebook, with design docs and a recorded demo.

## Tech Stack

- **Language:** Python 3.x
- **Data / ML:** scikit-learn (Random Forest, SVR, Linear/Polynomial regression, GridSearchCV), pandas, NumPy
- **Visualization:** matplotlib, seaborn, Streamlit's built-in charts
- **App / UI:** Streamlit (custom CSS)
- **Tooling:** Jupyter Notebook, VS Code Dev Container / GitHub Codespaces

## Getting Started

### Prerequisites
- Python 3.x (3.11 is what the dev container uses)
- pip

### Installation
```bash
git clone https://github.com/DCode-v05/Flight-Price-Prediction.git
cd Flight-Price-Prediction
pip install -r requirements.txt
```

`requirements.txt` lists pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn and streamlit.

### Running
```bash
# launch the web app (loads the pre-trained model)
streamlit run app.py
```

The app opens on `http://localhost:8501`. To explore the notebook or retrain the models:

```bash
jupyter notebook Price_Detection.ipynb
```

The notebook reads its CSVs from `/content/` (the Colab layout), so change those paths to `train/Data_Train.csv` and `test/Test_set.csv` before running it locally.

If you open the repo in a GitHub Codespace, the dev container installs the requirements and starts the Streamlit server for you.

## Usage

1. Run `streamlit run app.py`.
2. Pick your **airline**, **origin** and **destination** from the dropdowns.
3. Choose a **booking date** (today or later), set the **ticket type** (Economy / Business), enter **adult** and **child** counts, and set your **max number of stops**.
4. Optionally pick an **additional info** tag (meal, baggage, layover, etc.).
5. Hit **Predict** to get the estimated total fare (per passenger, with the business surcharge and last-minute surge applied where they apply) and a chart comparing price across airlines.

The notebook is where to look for the EDA, the model comparison, the correlation heatmap, the prediction-accuracy plots, or to retrain and re-export the model.

## Project Structure

```
Flight-Price-Prediction/
├── app.py                    # Streamlit app: form, encoding, prediction, pricing logic, chart
├── Price_Detection.ipynb     # EDA, feature engineering, model comparison, GridSearchCV tuning
├── requirements.txt          # pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, streamlit
├── .gitignore
├── model/
│   └── best_model.pkl        # Plain Random Forest, persisted (~15 MB)
├── train/
│   ├── Data_Train.csv        # Training data
│   └── Data_Train.xlsx
├── test/
│   ├── Test_set.csv          # Test data
│   └── Test_set.xlsx
├── Sample_submission.csv     # Expected output format
├── Sample_submission.xlsx
├── docs/
│   ├── Flight_Arch.docx      # Architecture document
│   ├── Flight_HLD.docx       # High-level design
│   ├── Flight_LLD.docx       # Low-level design
│   ├── Flight_WF.docx        # Workflow document
│   ├── Flight_Pro_Rep.pptx   # Project report deck
│   └── Flight Prediction.mp4 # Demo video
├── .devcontainer/
│   └── devcontainer.json     # Codespaces config (auto-runs Streamlit on :8501)
└── README.md
```

---

## Contact

<table>
  <tr><td><b>Portfolio:</b> <a href="https://www.denistan.me">Denistan</a></td><td><b>LinkedIn:</b> <a href="https://www.linkedin.com/in/denistanb">denistanb</a></td></tr>
  <tr><td><b>GitHub:</b> <a href="https://github.com/DCode-v05">DCode-v05</a></td><td><b>LeetCode:</b> <a href="https://leetcode.com/u/Denistan_B">Denistan_B</a></td></tr>
  <tr><td colspan="2" align="center"><b>Email:</b> <a href="mailto:denistanb05@gmail.com">denistanb05@gmail.com</a></td></tr>
</table>

Made with ❤️ by **Denistan B**
