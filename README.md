# Support Ticket Priority Prediction

An end-to-end machine learning project that predicts the priority of customer support tickets as **Low**, **Medium**, or **High**.

The project covers the complete machine learning workflow, including:

* Data cleaning and preprocessing
* Feature selection
* Exploratory data analysis
* Model training and comparison
* Cross-validation
* Model evaluation
* Model serialization
* FastAPI model serving
* Docker containerization
* Git/GitHub version control

---

## Project Objective

Customer support teams may receive thousands of tickets with different levels of urgency.

Manually assigning priority to every ticket can be slow and inconsistent.

The goal of this project is to automatically classify a support ticket into one of three priority levels:

* **Low**
* **Medium**
* **High**

based on customer, incident, system, and business-related information.

---

## Dataset

The original dataset contains:

* **50,000 support ticket records**
* **33 columns**
* **3 target classes**

Dataset shape:

```text
50,000 rows × 33 columns
```

The full dataset is used locally for model development, training, and evaluation.

Because the complete dataset is not included in the GitHub repository, a smaller sample containing **100 rows** is provided for demonstration and exploratory purposes.

```text
data/
├── support_tickets_100.csv
└── Support_tickets.csv   # Full dataset used locally
```

The full dataset is excluded from GitHub through `.gitignore`.

### Target Distribution

| Priority  |    Tickets | Percentage |
| --------- | ---------: | ---------: |
| Low       |     25,000 |        50% |
| Medium    |     17,500 |        35% |
| High      |      7,500 |        15% |
| **Total** | **50,000** |   **100%** |

The target distribution is imbalanced, especially for the `High` class.

For this reason, model evaluation focuses not only on accuracy but also on:

* Precision
* Recall
* F1 Score
* Macro F1 Score
* Cross-validation performance

---

## Features Used

Although the original dataset contains **33 columns**, the final machine learning model uses **20 selected input features**.

### Numerical Features

1. `customers_affected`
2. `downtime_min`
3. `error_rate_pct`
4. `org_users`
5. `past_30d_tickets`
6. `past_90d_incidents`
7. `payment_impact_flag`
8. `security_incident_flag`
9. `data_loss_flag`
10. `has_runbook`
11. `description_length`

### Categorical Features

1. `company_size`
2. `industry`
3. `customer_tier`
4. `region`
5. `product_area`
6. `booking_channel`
7. `reported_by_role`
8. `day_of_week`
9. `customer_sentiment`

### Target

```text
priority
```

---

## Features Excluded from Modeling

Identifier columns and redundant encoded columns are not used as model inputs.

Examples include:

```text
ticket_id
company_id
priority_cat
industry_cat
customer_tier_cat
region_cat
product_area_cat
booking_channel_cat
reported_by_role_cat
customer_sentiment_cat
```

The categorical encoding is performed directly inside the Scikit-learn preprocessing pipeline.

---

## Data Cleaning

The training workflow performs basic data cleaning before model training.

### Cleaning Steps

* Clean column names
* Remove duplicate rows
* Standardize categorical text values
* Convert numerical features to numeric data types
* Handle invalid numerical values
* Remove observations with missing target values
* Validate required model features

---

## Preprocessing

The project uses a Scikit-learn `ColumnTransformer` to apply different preprocessing steps to numerical and categorical variables.

### Numerical Features

Missing numerical values are filled using:

```text
Median Imputation
```

Pipeline:

```text
Numerical Data
      ↓
Median Imputation
      ↓
Machine Learning Model
```

### Categorical Features

Missing categorical values are filled using the most frequent value.

Categorical variables are then converted using:

```text
OneHotEncoder
```

Pipeline:

```text
Categorical Data
       ↓
Most Frequent Imputation
       ↓
One-Hot Encoding
       ↓
Machine Learning Model
```

The encoder also supports previously unseen categories using:

```python
handle_unknown="ignore"
```

---

## Train/Test Split

The dataset is divided into:

```text
80% Training Data
20% Testing Data
```

A stratified split is used to preserve the original Low, Medium, and High priority distribution.

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

---

## Machine Learning Models

Two ensemble machine learning algorithms were trained and compared.

### 1. Random Forest Classifier

Random Forest was used as the baseline ensemble model.

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)
```

`class_weight="balanced"` helps the model account for the unequal distribution of the target classes.

---

### 2. Gradient Boosting Classifier

The second model was Gradient Boosting.

```python
GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
```

Gradient Boosting achieved substantially better validation and test performance than Random Forest.

---

## Cross-Validation

Both models are evaluated using:

```text
5-Fold Stratified Cross-Validation
```

The primary model-selection metric is:

```text
Macro F1 Score
```

Macro F1 gives equal importance to all three classes, making it more appropriate than accuracy alone for this imbalanced classification problem.

---

## Model Comparison

### Random Forest

```text
Cross-Validation Macro F1 ≈ 0.770
```

### Gradient Boosting

```text
Cross-Validation Macro F1 ≈ 0.929
```

Gradient Boosting clearly outperformed Random Forest and was automatically selected as the final model.

---

## Final Model Performance

### Best Model

**Gradient Boosting Classifier**

| Metric                    |     Score |
| ------------------------- | --------: |
| Test Accuracy             | **93.4%** |
| Test Macro F1             | **92.3%** |
| Cross-Validation Macro F1 | **92.9%** |

---

## Classification Report

| Priority | Precision | Recall | F1-Score |
| -------- | --------: | -----: | -------: |
| High     |      0.97 |   0.83 |     0.90 |
| Low      |      0.96 |   0.97 |     0.96 |
| Medium   |      0.89 |   0.93 |     0.91 |

The final model achieved strong performance across all three priority classes.

---

## Machine Learning Workflow

```text
50,000 Support Tickets
          ↓
      Data Cleaning
          ↓
    Feature Selection
          ↓
      Preprocessing
          ↓
     Train/Test Split
          ↓
 ┌───────────────────────┐
 │     Random Forest     │
 │                       │
 │   Gradient Boosting   │
 └───────────────────────┘
          ↓
5-Fold Cross-Validation
          ↓
  Compare Macro F1
          ↓
 Select Best Model
          ↓
  Test Set Evaluation
          ↓
 Retrain on Full Dataset
          ↓
    Save Model
          ↓
       FastAPI
          ↓
        Docker
```

---

## Model Serialization

After evaluating the candidate models, the best-performing model is retrained using all available data.

The complete preprocessing + machine learning pipeline is then saved using `joblib`.

```text
models/support_ticket_priority_model.joblib
```

The saved object contains:

```text
Preprocessing Pipeline
        +
Gradient Boosting Model
```

This allows new raw ticket data to pass through exactly the same preprocessing pipeline used during training.

The generated `.joblib` file is excluded from GitHub and can be recreated by running the training script.

---

## Project Structure

```text
support-ticket-priority-prediction/
│
├── app/
│   ├── __init__.py
│   └── main.py
│
├── data/
│   ├── .gitkeep
│   └── support_tickets_100.csv
│
├── models/
│   └── .gitkeep
│
├── notebook/
│   └── model_exploration.ipynb
│
├── src/
│   ├── eda.py
│   └── train.py
│
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Behnushkiani/support-ticket-priority-prediction.git
```

Go to the project directory:

```bash
cd support-ticket-priority-prediction
```

Install the required packages:

```bash
python3 -m pip install -r requirements.txt
```

---

## Train the Model

The full dataset should be placed at:

```text
data/Support_tickets.csv
```

Then run:

```bash
python3 src/train.py
```

The training script will automatically:

1. Load the dataset
2. Clean the data
3. Select the required features
4. Preprocess numerical features
5. Preprocess categorical features
6. Split the data into training and test sets
7. Train Random Forest
8. Train Gradient Boosting
9. Perform 5-fold cross-validation
10. Compare Macro F1 scores
11. Select the best model
12. Evaluate the selected model
13. Retrain the best model using all available data
14. Save the final model

The generated model will be saved as:

```text
models/support_ticket_priority_model.joblib
```

---

## FastAPI Model Serving

The trained machine learning model is exposed through a REST API using **FastAPI**.

Start the API:

```bash
python3 -m uvicorn app.main:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### Home

```text
GET /
```

Checks whether the API is running.

---

### Health Check

```text
GET /health
```

Example:

```json
{
  "status": "healthy"
}
```

---

### Predict Ticket Priority

```text
POST /predict
```

The endpoint accepts support-ticket information and returns the predicted priority.

---

## Example Prediction Request

```json
{
  "customers_affected": 1500,
  "downtime_min": 120,
  "error_rate_pct": 35,
  "org_users": 5000,
  "past_30d_tickets": 25,
  "past_90d_incidents": 8,
  "payment_impact_flag": 1,
  "security_incident_flag": 1,
  "data_loss_flag": 0,
  "has_runbook": 0,
  "description_length": 350,
  "company_size": "large",
  "industry": "finance",
  "customer_tier": "enterprise",
  "region": "north america",
  "product_area": "payments",
  "booking_channel": "web",
  "reported_by_role": "admin",
  "day_of_week": "monday",
  "customer_sentiment": "negative"
}
```

Example response:

```json
{
  "predicted_priority": "high"
}
```

---

## Docker

The application can be packaged and run inside a Docker container.

Before building the Docker image, first train the model so that:

```text
models/support_ticket_priority_model.joblib
```

exists locally.

### Build Docker Image

```bash
docker build -t support-ticket-api .
```

### Run Docker Container

```bash
docker run -p 8000:8000 support-ticket-api
```

Then open:

```text
http://127.0.0.1:8000/docs
```

---

## Technology Stack

### Data & Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* Random Forest
* Gradient Boosting
* Joblib

### API

* FastAPI
* Pydantic
* Uvicorn

### Deployment

* Docker

### Development

* Jupyter Notebook
* VS Code
* Git
* GitHub

---

## Skills Demonstrated

This project demonstrates practical experience with:

* Exploratory Data Analysis
* Data Cleaning
* Feature Selection
* Missing Value Handling
* Categorical Encoding
* Machine Learning Pipelines
* Classification
* Class Imbalance
* Random Forest
* Gradient Boosting
* Model Comparison
* Stratified Train/Test Split
* Cross-Validation
* Precision
* Recall
* F1 Score
* Macro F1
* Model Evaluation
* Model Serialization
* REST API Development
* FastAPI
* Docker
* Git
* GitHub

---

## Future Improvements

Potential future improvements include:

* XGBoost
* LightGBM
* Hyperparameter tuning with Optuna
* GridSearchCV
* SHAP model explainability
* Feature importance analysis
* Confusion matrix visualization
* Automated API testing
* MLflow experiment tracking
* Cloud deployment
* API logging
* Model monitoring
* Data drift monitoring
* Automated model retraining

---

## Author

**Behnush Kiani**

Machine Learning | AI | Data Science

GitHub: [Behnushkiani](https://github.com/Behnushkiani)
