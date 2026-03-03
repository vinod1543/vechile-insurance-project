<div align="center">

# 🚗 Vehicle Insurance Cross-Sell Prediction
### An End-to-End MLOps Project

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?style=for-the-badge&logo=mongodb&logoColor=white)](https://mongodb.com)
[![AWS](https://img.shields.io/badge/AWS-Cloud-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI/CD-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)

<br/>

> **Predict whether a health insurance customer will also be interested in vehicle insurance** — productionized with a full MLOps stack: automated pipelines, cloud storage, containerized deployment, and CI/CD.

</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [ML Pipeline Components](#-ml-pipeline-components)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Cloud & Deployment Setup](#-cloud--deployment-setup)
- [CI/CD Pipeline](#-cicd-pipeline)
- [API Endpoints](#-api-endpoints)
- [Features at a Glance](#-features-at-a-glance)

---

## 🎯 Project Overview

This project tackles a **real-world binary classification problem**: predicting customer interest in vehicle insurance cross-sell opportunities. An insurance company needs to know — of its existing health insurance customers — which ones would be interested in purchasing vehicle insurance too.

The project is built as a **production-grade MLOps system** covering every stage from raw data in a NoSQL database to a live, auto-scaling web application deployed on AWS.

### Business Problem
Given customer demographics and vehicle information, predict: **Will this customer respond positively to a vehicle insurance offer?**

### Input Features

| Feature | Description |
|---|---|
| `Gender` | Customer gender |
| `Age` | Customer age |
| `Driving_License` | Does customer hold a driving license? |
| `Region_Code` | Region of the customer |
| `Previously_Insured` | Was vehicle previously insured? |
| `Annual_Premium` | Annual insurance premium amount |
| `Policy_Sales_Channel` | Channel through which policy was sold |
| `Vintage` | Number of days associated with the company |
| `Vehicle_Age` | Age of vehicle (< 1 Year / 1–2 Years / > 2 Years) |
| `Vehicle_Damage` | Was vehicle damaged in the past? |

**Target:** `Response` — 1 (Interested) / 0 (Not Interested)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                                     │
│                  MongoDB Atlas (Cloud NoSQL DB)                         │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       TRAINING PIPELINE                                 │
│                                                                         │
│  Data Ingestion → Data Validation → Data Transformation → Model Trainer │
│                                          │                              │
│                                    Model Evaluation ──→ Model Pusher   │
└───────────────────────────────────────────────────────┬─────────────────┘
                                                        │
                                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         MODEL REGISTRY                                  │
│                      AWS S3 Bucket (us-east-1)                          │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       SERVING LAYER                                     │
│             FastAPI App  +  Prediction Pipeline                         │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     DEPLOYMENT / CI-CD                                  │
│  GitHub Actions → Docker → AWS ECR → AWS EC2 (Self-Hosted Runner)      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ ML Pipeline Components

The training pipeline is fully automated and modular. Each component is independently configurable via dataclasses in `config_entity.py` and produces typed artifacts tracked in `artifact_entity.py`.

### 1. 📥 Data Ingestion
- Connects to **MongoDB Atlas** using a secure environment-variable-based URL
- Fetches raw data from the `Proj1-Data` collection in the `Proj1` database
- Converts key-value MongoDB documents → structured Pandas DataFrame
- Performs stratified **train/test split** (75% / 25%)
- Saves raw CSV files to a timestamped artifact directory

### 2. ✅ Data Validation
- Validates dataset schema against `config/schema.yaml`
- Checks column presence, data types, and statistical distributions
- Generates a detailed `report.yaml` with drift detection results
- Gates the pipeline — downstream components only run if validation passes

### 3. 🔄 Data Transformation
- Encodes categorical variables (`Gender`, `Vehicle_Age`, `Vehicle_Damage`)
- Creates one-hot encoded dummy columns
- Applies column renaming for ML-ready feature names
- Serializes the fitted preprocessor as `preprocessing.pkl` (reused at inference)
- Exports transformed data as `.npy` arrays

### 4. 🤖 Model Training
- Trains a **Random Forest Classifier** with tuned hyperparameters:
  - `n_estimators = 200`
  - Configurable `max_depth`, `min_samples_split`, `criterion`, `random_state`
- Evaluates against F1 Score, Precision, Recall, and Accuracy
- Saves the trained model wrapped in a custom `MyModel` estimator (combines preprocessor + model)
- Enforces a minimum expected accuracy threshold before accepting the model

### 5. 📊 Model Evaluation
- Loads the **current production model** from AWS S3 (model registry)
- Compares challenger model vs. production model on held-out test data using **F1 Score**
- Accepts challenger only if improvement exceeds configured threshold (`0.02` by default)
- Zero-downtime model promotion strategy

### 6. 🚀 Model Pusher
- Pushes accepted models to **AWS S3** bucket (`my-model-mlopsproj`)
- Maintains a model registry under the `model-registry/` S3 key prefix
- Enables seamless model serving via `Proj1Estimator` (S3-backed estimator class)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10 |
| **ML Framework** | scikit-learn (RandomForestClassifier) |
| **Web Framework** | FastAPI + Uvicorn |
| **Frontend** | Jinja2 Templates + Custom CSS |
| **Database** | MongoDB Atlas (Cloud) |
| **Object Storage / Model Registry** | AWS S3 |
| **Containerization** | Docker (python:3.10-slim-buster) |
| **Container Registry** | AWS ECR (Elastic Container Registry) |
| **Compute** | AWS EC2 (Ubuntu Server 24.04, T2 Medium) |
| **CI/CD** | GitHub Actions (Self-Hosted Runner on EC2) |
| **Configuration** | YAML-based schema & model config |
| **Logging** | Custom structured logger (`src/logger`) |
| **Exception Handling** | Custom exception module (`src/exception`) |
| **Package Management** | pip + `pyproject.toml` / `setup.py` |

---

## 📁 Project Structure

```
vehicle-insurance-project/
│
├── .github/
│   └── workflows/
│       └── aws.yaml              # GitHub Actions CI/CD workflow
│
├── config/
│   ├── model.yaml                # Model hyperparameter configuration
│   └── schema.yaml               # Dataset schema for validation
│
├── notebook/
│   ├── data.csv                  # Raw dataset
│   ├── exp-notebook.ipynb        # EDA & Feature Engineering notebook
│   └── mongoDB_demo.ipynb        # MongoDB data push demo
│
├── src/
│   ├── cloud_storage/
│   │   └── aws_storage.py        # S3 upload/download helpers
│   │
│   ├── components/               # Core ML pipeline stages
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   │
│   ├── configuration/
│   │   ├── aws_connection.py     # AWS S3 session manager
│   │   └── mongo_db_connection.py
│   │
│   ├── constants/
│   │   └── __init__.py           # All project-wide constants
│   │
│   ├── data_access/
│   │   └── proj1_data.py         # MongoDB → DataFrame bridge
│   │
│   ├── entity/
│   │   ├── config_entity.py      # Pipeline configuration dataclasses
│   │   ├── artifact_entity.py    # Pipeline artifact dataclasses
│   │   ├── estimator.py          # Custom MyModel estimator wrapper
│   │   └── s3_estimator.py       # S3-backed estimator for inference
│   │
│   ├── exception/                # Custom exception handling
│   ├── logger/                   # Structured logging
│   ├── pipeline/
│   │   ├── training_pipeline.py  # Orchestrates full training flow
│   │   └── prediction_pipeline.py
│   └── utils/
│       └── main_utils.py         # Serialization & utility helpers
│
├── static/css/style.css          # Frontend styling
├── templates/vehicledata.html    # Prediction form template
├── app.py                        # FastAPI application entry point
├── Dockerfile                    # Container definition
├── requirements.txt
├── setup.py
└── pyproject.toml
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10
- MongoDB Atlas account (free tier works)
- AWS account (free tier for S3; EC2 is chargeable)
- Docker (for containerized deployment)
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/vehicle-insurance-project.git
cd vehicle-insurance-project
```

### 2. Create and Activate Virtual Environment

```bash
# Using conda (recommended)
conda create -n vehicle python=3.10 -y
conda activate vehicle

# OR using venv
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip list  # Verify local packages (src.*) are installed
```

### 4. Configure Environment Variables

```bash
# PowerShell
$env:MONGODB_URL = "mongodb+srv://<username>:<password>@cluster.mongodb.net/"
$env:AWS_ACCESS_KEY_ID = "your-access-key-id"
$env:AWS_SECRET_ACCESS_KEY = "your-secret-access-key"

# Bash / Linux / macOS
export MONGODB_URL="mongodb+srv://<username>:<password>@cluster.mongodb.net/"
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
```

### 5. Run the Application

```bash
python app.py
```

Visit `http://localhost:5000` to access the prediction UI.  
Trigger model training at `http://localhost:5000/training`.

---

## ☁️ Cloud & Deployment Setup

### MongoDB Atlas
1. Create a free cluster on [MongoDB Atlas](https://cloud.mongodb.com)
2. Create a database user and whitelist IP `0.0.0.0/0`
3. Copy the connection string (Python driver, v3.6+) and set it as `MONGODB_URL`
4. Push data using `notebook/mongoDB_demo.ipynb`

### AWS Services

| Service | Purpose | Configuration |
|---|---|---|
| **IAM** | Secure programmatic access | `AdministratorAccess` policy; access key exported as env vars |
| **S3** | Model registry | Bucket: `my-model-mlopsproj`, Region: `us-east-1` |
| **ECR** | Docker image registry | Repo: `vehicleproj` |
| **EC2** | Production server | Ubuntu 24.04, T2 Medium, port 5000 exposed |

---

## 🔄 CI/CD Pipeline

Every `git push` to `main` automatically triggers a two-stage GitHub Actions workflow:

```
Push to main
     │
     ▼
┌─────────────────────────┐
│  Continuous Integration │
│  (runs on: ubuntu-latest)│
│                         │
│  1. Checkout code       │
│  2. Configure AWS creds │
│  3. Login to ECR        │
│  4. Docker build & push │
│     → ECR Registry      │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Continuous Deployment   │
│ (runs on: self-hosted   │
│  EC2 runner)            │
│                         │
│  1. Pull latest image   │
│  2. docker run with     │
│     injected env vars   │
│  3. App live on :5000   │
└─────────────────────────┘
```

### GitHub Secrets Required

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret |
| `AWS_DEFAULT_REGION` | `us-east-1` |
| `ECR_REPO` | ECR repository name |
| `MONGODB_URL` | MongoDB Atlas connection string |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Renders the vehicle data input form |
| `POST` | `/predict` | Accepts form data, returns prediction |
| `GET` | `/training` | Triggers the full training pipeline |

---

## ✨ Features at a Glance

```
✅  Modular ML pipeline with typed config & artifact dataclasses
✅  NoSQL data source — MongoDB Atlas
✅  Automated schema validation with drift reporting
✅  Custom preprocessor serialization (preprocessing.pkl)
✅  Champion-challenger model evaluation before deployment
✅  Cloud-native model registry on AWS S3
✅  Custom S3-backed estimator for zero-config inference
✅  FastAPI REST API with CORS support
✅  Jinja2-powered frontend with custom CSS
✅  Fully Dockerized application (python:3.10-slim-buster)
✅  AWS ECR for private Docker image storage
✅  AWS EC2 self-hosted GitHub Actions runner
✅  Automated CI/CD on every push to main branch
✅  Environment-variable-driven secrets management
✅  Timestamped artifact versioning per pipeline run
✅  Structured logging throughout all components
✅  Custom exception hierarchy with traceback chaining
```

---

<div align="center">

### Built with a focus on production readiness, reproducibility & scalability.

**Data → Train → Validate → Push → Serve → Repeat — fully automated.**

</div>
