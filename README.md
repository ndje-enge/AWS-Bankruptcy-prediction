# Bankruptcy Prediction Model - AWS SageMaker Deployment

Multi-Layer Perceptron Classifier model for Taiwanese company bankruptcy prediction deployed on AWS SageMaker + REST API. Personal project to understand AWS deployment and cost optimization.


## Dataset

The dataset contains 95 financial features from Taiwanese companies:
- **Financial ratios** : ROA, ROE, liquidity ratios
- **Performance indicators** : Margins, efficiency ratios
- **Liquidity metrics** : Cash ratios, turnover
- **Profitability data** : Profits, revenues, costs
- **Structure indicators** : Debt, equity, assets

## Feature Engineering

### 1. Feature Selection (SelectKBest)
- **Method** : Univariate F-test (f_classif)
- **Criterion** : Selection of the 50 best features out of 95
- **Objective** : Reduce dimensionality and eliminate non-informative features
- **Benefits** : Improved performance and reduced training time

### 2. Data Balancing (SMOTE-Tomek)
- **Initial problem** : Class imbalance (6599 non-bankrupt vs 220 bankrupt)
- **Solution** : SMOTE-Tomek (combination of SMOTE and Tomek Links)
  - **SMOTE** : Generation of synthetic samples for the minority class
  - **Tomek Links** : Removal of ambiguous samples at class boundaries
- **Result** : 13,176 balanced samples (6588 per class)

### 3. Data Normalization
- **Method** : StandardScaler (Z-score normalization)
- **Formula** : (x - μ) / σ
- **Objective** : Standardize features for the MLPClassifier algorithm
- **Benefits** : Faster convergence and better performance

## Local Training

### MLPClassifier Model Architecture
```python
MLPClassifier(
    hidden_layer_sizes=(100, 50),    # 2 hidden layers architecture
    activation='relu',               # ReLU activation function
    solver='adam',                   # Adam optimizer
    alpha=0.001,                     # L2 regularization
    learning_rate='adaptive',        # Adaptive learning rate
    learning_rate_init=0.001,        # Initial learning rate
    max_iter=1000,                   # Maximum number of iterations
    early_stopping=True,             # Early stopping
    validation_fraction=0.1,         # 10% of data for validation
    n_iter_no_change=10             # Patience for early stopping
)
```

### Training Process
1. **Data split** : 80% training, 20% test
2. **Normalization** : StandardScaler on training data
3. **Training** : MLPClassifier with early stopping
4. **Validation** : Using 10% of training data for validation
5. **Evaluation** : Metrics on test set

### Applied Optimizations
- **L2 Regularization** : Prevent overfitting (alpha=0.001)
- **Early stopping** : Avoid overfitting with patience of 10 iterations
- **Adaptive learning rate** : Automatic learning rate adjustment
- **Optimized architecture** : 2 hidden layers (100, 50)

## Deployment Architecture

### API REST (Lambda + API Gateway)
```
Client → API Gateway → Lambda Function → SageMaker Endpoint
```

## Deployed Endpoints

### REST API (Lambda + API Gateway)
- **URL** : `https://sv6rnbh9mi.execute-api.eu-west-3.amazonaws.com/prod/predict`
- **Method** : POST
- **Format** : JSON
- **CORS** : Enabled
- **Lambda Function** : `bankruptcy-prediction-api`
- **Status** : On demand

### Direct SageMaker Endpoint
- **Name** : `SAGEMAKER_ENDPOINT_NAME`
- **Region** : eu-west-3
- **Instance** : ml.t2.medium
- **Format** : JSON
- **Status** : On demand 


## Usage

### REST API 

The API is exposed via Lambda + API Gateway for simpler and more secure usage.


#### Usage with curl
```bash
curl -X POST https://sv6rnbh9mi.execute-api.eu-west-3.amazonaws.com/prod/predict \
  -H 'Content-Type: application/json' \
  -d '{"data": [0.1, 0.2, -0.05, 0.15, ...]}'
```

#### Usage with Python
```python
import requests
import json

api_url = "https://sv6rnbh9mi.execute-api.eu-west-3.amazonaws.com/prod/predict"

# Input data (50 features)
data = [0.1] * 50

# Make prediction
response = requests.post(
    api_url,
    json={"data": data},
    headers={'Content-Type': 'application/json'}
)

result = response.json()
print(result)
```


### Response Format

```json
{
  "prediction": 0,
  "probability": {
    "not_bankrupt": 1.0,
    "bankrupt": 4.4123744484191364e-20
  },
  "risk_level": "low"
}
```

## Performance and Results

### Model Metrics
- **ROC-AUC Score** : 0.92
- **Accuracy** : 0.93
- **Precision** : 0.91
- **Training time** : 2-3 minutes (local)
- **Features used** : 50 (selected by SelectKBest)



## Dependencies

### Python Packages
- **boto3** : AWS SDK for Python
- **sagemaker** : AWS SageMaker SDK
- **scikit-learn** : Machine Learning (MLPClassifier, SelectKBest, StandardScaler)
- **pandas** : Data manipulation
- **numpy** : Numerical computations
- **imbalanced-learn** : Imbalanced data handling (SMOTE-Tomek)
- **joblib** : Model serialization
- **requests** : HTTP requests for testing

### AWS Services
- **SageMaker** : ML model deployment
- **Lambda** : Serverless function for API
- **API Gateway** : REST API exposure
- **S3** : Model artifacts storage
- **IAM** : Permissions management

## Steps to Complete Training and Deploy Pipeline

### 1. Local Training
```bash
# Step 1: Train the model locally
python3 train_model.py
```

**This script performs:**
- Data loading (6819 samples, 95 features)
- Selection of the 50 best features with SelectKBest
- Data balancing with SMOTE-Tomek (13,176 samples)
- Train/test split (80%/20%)
- Normalization with StandardScaler
- MLPClassifier training with early stopping
- Saving artifacts in the `models/` folder

### 2. AWS SageMaker Deployment
```bash
# Step 2: Deploy the model on SageMaker
python3 deploy.py
```

**This script performs:**
- Model package creation (model + scaler + features + inference script)
- Compression into `optimized_compatible_model.tar.gz` archive
- Upload to S3 (`bankruptcy-prediction-models`)
- SageMaker model creation
- Endpoint deployment with `ml.t2.medium` instance
- Automatic endpoint testing

### 3. Lambda + API Gateway Deployment
```bash
# Step 3: Deploy the REST API
python3 deploy_lambda_api.py
```

**This script performs:**
- IAM role creation for Lambda
- Lambda function deployment
- API Gateway creation
- Integration configuration
- Permission assignment
- Complete API testing

### 4. API Testing
```bash
# Step 4: Test the REST API
python3 test_lambda_api.py
```

## Configuration 

### AWS Configuration

1. **Create the configuration file** :
```bash
cp config_example.py config.py
```

2. **Modify `config.py`** with your AWS information :
```python
# SageMaker Configuration
SAGEMAKER_ROLE_ARN = "arn:aws:iam::YOUR_ACCOUNT:role/YOUR_SAGEMAKER_ROLE"
SAGEMAKER_MODEL_NAME = "your-model-name"

# S3 Configuration
S3_BUCKET_NAME = "your-bucket-name"

# Lambda Configuration
LAMBDA_FUNCTION_NAME = "your-lambda-function-name"

# API Gateway Configuration
API_GATEWAY_ID = "your-api-gateway-id"
API_GATEWAY_URL = "https://your-api-gateway-id.execute-api.region.amazonaws.com/prod/predict"

# AWS Region
AWS_REGION = "your-region"
```

### AWS Prerequisites

- **AWS Account** with appropriate permissions
- **SageMaker Role** with S3 access
- **S3 Bucket** to store artifacts
- **IAM Permissions** to create Lambda roles


## Project Management (Pause/Resume)

The `manage_project.py` script allows easy management of AWS resources to optimize costs.

### Available Commands

#### Check status
```bash
python3 manage_project.py status
```
Displays the status of all resources (SageMaker, Lambda, API Gateway).

#### Pause
```bash
python3 manage_project.py pause
```
- Removes SageMaker endpoint
- Keeps Lambda and API Gateway
- API remains accessible but without predictions

#### Resume project
```bash
python3 manage_project.py resume
```
- Recreates SageMaker endpoint
- Restores full API service
- Ready for demonstrations

#### Test API
```bash
python3 manage_project.py test
```
Tests the deployed API with sample data.



## Technical Deployment Details

### Lambda + API Gateway Architecture

#### Lambda Function (`lambda_function.py`)
The Lambda function acts as an intelligent proxy between API Gateway and SageMaker:

**Features:**
- **Data validation** : Format and feature count verification (50)
- **CORS handling** : Cross-origin request support for web applications
- **SageMaker invocation** : Endpoint calling with error handling
- **Response enrichment** : Metadata addition (confidence, timestamp, model info)
- **Error handling** : Structured error returns with appropriate HTTP codes

**Processing flow:**
1. HTTP request reception from API Gateway
2. Input JSON parsing and validation
3. Data conversion to SageMaker format
4. SageMaker endpoint invocation
5. Response processing and enrichment
6. JSON response return to API Gateway

#### API Gateway Configuration
- **Resource** : `/predict`
- **Methods** : POST (prediction) + OPTIONS (CORS preflight)
- **Integration** : Lambda Proxy Integration
- **CORS** : Configured for all origins
- **Deployment** : "prod" stage with public URL

### SageMaker Inference Script
The file `inference.py` contains:
- **model_fn()** : Model and artifacts loading (scaler, features)
- **input_fn()** : Input JSON data parsing (50 features)
- **predict_fn()** : Prediction with normalization and scoring
- **output_fn()** : JSON response formatting


