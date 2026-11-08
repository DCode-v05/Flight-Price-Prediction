# Flight Price Prediction

Predicts Indian domestic flight fares from airline, route, date and stops, and serves the model through a small Streamlit app. The model is a random forest trained in `Price_Detection.ipynb`.

## How it works

**Notebook.** `Price_Detection.ipynb` loads `train/Data_Train.csv`, fills missing values, splits the journey date into day, month and year, turns the duration into hours, drops `Route`, `Dep_Time` and `Arrival_Time`, and encodes the categorical columns as integers. It then compares four model types on an 80/20 split:

| Model | R2 on test split |
|---|---|
| Linear regression | 0.38 |
| Polynomial regression (degree 2 / 3) | 0.57 / 0.36 |
| Random forest (`random_state=0`) | 0.85 |
| Random forest after `GridSearchCV` | 0.84 |
| SVR | not recorded in the notebook output |

The model saved to `model/best_model.pkl` is the plain random forest (`ran_model`), not the grid-searched one, since it scored slightly higher.

**App.** `app.py` loads the pickle and shows a form for airline, origin, destination, booking date, ticket type, adults, children, stops and additional info. After you press Predict it takes the model's per-passenger fare and applies rules written in the app: +25% for Business, multiplied by adults plus children, and doubled if the date is less than a day away. A bar chart of price by airline from the training data sits below.

The fare rules in the app are rough multipliers I added, not something learned from data.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the notebook with `jupyter notebook Price_Detection.ipynb`. It reads the CSVs from `/content/` (Colab), so change the paths to `train/Data_Train.csv` and `test/Test_set.csv`.

The `.devcontainer` config targets Python 3.11 and starts Streamlit in a Codespace.

## Known issue

The notebook model is trained on nine features including `Duration`. `app.py` sends eight columns and no `Duration`, so the prediction call may fail with a feature mismatch. I have not run the app to confirm.

## Layout

```
app.py                  Streamlit app
Price_Detection.ipynb   EDA, model comparison, tuning, export
model/best_model.pkl    Trained random forest (~15 MB)
train/                  Data_Train.csv, Data_Train.xlsx
test/                   Test_set.csv, Test_set.xlsx
Sample_submission.*     Expected submission format
docs/                   Architecture, HLD, LLD and workflow documents (.docx),
                        project report (.pptx) and a demo video (.mp4)
.devcontainer/          Codespaces setup
```
