# **Home Price Prediction Maching Learning Application**

A real-time, end-to-end machine learning application built with Flask and integrated with MLflow for tracking and model management. The application predicts house prices based on user input, leveraging trained regression models and providing a web interface for seamless interaction.

---

## **Project Overview**

This project is a **real-time, end-to-end machine learning application** that predicts property prices based on a set of features provided by the user. The pipeline includes all stages from data preprocessing, model training, prediction, and real-time web interface integration. It is built using Flask for the web app, machine learning models for prediction, and MLflow for experiment tracking and model management.

### **Key Components of the Project:**
- **Machine Learning Pipeline**: From data preprocessing and feature engineering to model training and prediction.
- **Flask Web Application**: Frontend interface where users can input property details and receive price predictions.
- **MLflow Integration**: Tracks experiments, models, and metrics for consistent version control and efficient model management.
- **Real-time Prediction**: Predicts house prices in real-time as users input property details.

---

## **Features**

1. **Real-Time House Price Prediction**:
   - Input property features (e.g., area, number of bedrooms, amenities) and get instant predictions.
   
2. **End-to-End Machine Learning Pipeline**:
   - Data ingestion, cleaning, feature engineering, model training, and prediction.
   - Predictive models trained using advanced regression techniques such as Random Forest, Gradient Boosting, and ElasticNet.

3. **MLflow Experiment Tracking**:
   - Logs experiment details including hyperparameters, metrics, and models.
   - Facilitates model versioning and comparison for tracking model performance over time.
   
4. **Web Interface**:
   - **Input Form**: Users can fill in property details such as area, price per square foot, number of bedrooms, and more.
   - **Dynamic Results**: Predictions are displayed on a separate results page.
   
5. **Model Retraining**:
   - The system allows retraining the model on-demand using the `/train` route.

---

## **System Requirements**

- **Python**: 3.8+
- **Dependencies**: 
  - `flask`
  - `numpy`
  - `pandas`
  - `sklearn`
  - `mlflow`
- **Web Browser**: Chrome, Firefox, or any modern browser.

---

## **Real-Time End-to-End Machine Learning Pipeline**
**Data Preprocessing:**
- Includes normalization of numerical features and encoding categorical variables.
- Cleans and prepares the data for model training.
**Model Training:**
- Models include Random Forest, Gradient Boosting, and ElasticNet regression.
- Hyperparameter tuning via GridSearchCV for optimizing model performance.
**Prediction Pipeline:**
- Takes user input via the web interface and returns predicted property prices based on the trained model.
- MLflow Integration
**Experiment Tracking:**
- Logs parameters, metrics (e.g., R², MSE), and artifacts during model training.
- Facilitates comparison between different models and configurations.
**Model Versioning:**
- Allows users to track and compare multiple versions of the model in the MLflow registry.
**Model Deployment:**
- MLflow makes it easy to transition from experimentation to production-ready models.
- Frontend (Web Interface)
**HTML Templates:**
- index.html: Renders input form for property details.
- results.html: Displays predictions based on user input.
**Dynamic Routing:**
- /predict: Handles POST requests, processes user input, and displays the prediction.
- /train: Triggers the retraining of the model and logs new experiments.

---

## **Setup Instructions**

### **1. Clone the Repository**
- git clone https://github.com/anu0408/HousePricePrediction.git
- cd HousePricePrediction
### 2. Install Dependencies**
- Install all required Python packages using pip:
- pip install -r requirements.txt
### **3. Set Up MLflow**
- Start the MLflow tracking server (if using local server):
- mlflow ui
- Access the MLflow UI at http://localhost:5000.
- Configure a remote tracking server by setting the MLFLOW_TRACKING_URI environment variable:
- export MLFLOW_TRACKING_URI="http://your-remote-server"
### **4. Run the Application**
- Start the Flask application:
- python app.py
- Access the application at http://localhost:8080.

### Usage
**1. Predict Property Price**
Open the web app in your browser.
Fill out the form with details such as:
   - Property area
    - Price per square foot
   - Number of bedrooms
   - Available amenities (e.g., pool, park, gym, etc.)
   - Submit the form to get the predicted house price instantly.
**2. Retrain the Model**
To retrain the model:
Navigate to the /train route in your browser:
http://localhost:8080/train
This will trigger the training pipeline and update the model, with experiments tracked in MLflow.
**3. Track Experiments with MLflow**
Access the MLflow UI at http://localhost:5000 to monitor training experiments.
View metrics, parameters, and models.
Compare experiments to evaluate model performance.



### Future Improvements
- Enhanced Model Accuracy: Incorporate more sophisticated machine learning models and fine-tune hyperparameters further.
- MLflow Automation: Use MLflow's deployment capabilities for seamless model deployment to production environments.
- Database Integration: Add support for persistent data storage using relational or NoSQL databases.
- Cloud Deployment: Host the application and MLflow on cloud services such as AWS, GCP, or Azure.



























```markdown
# Home-Price-Prediction

## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the app.py

# How to run?
### STEPS:


### STEP 01 - Create a conda environment after opening the repository

```bash
conda create -n homePrice python=3.8 -y
```

```bash
conda activate homePrice
```

### STEP 02 - Install the requirements
```bash
pip install -r requirements.txt
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up your local host and port
```

## MLflow

[Documentation](https://mlflow.org/docs/latest/index.html)

##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=*** \
MLFLOW_TRACKING_USERNAME=sekhar.pogula \
MLFLOW_TRACKING_PASSWORD=*** \
conda env config vars list  
conda env config vars set MLFLOW_TRACKING_USERNAME=sekhar.pogula

python script.py

Run this to export as env variables:

```bash
export MLFLOW_TRACKING_URI=***
export MLFLOW_TRACKING_USERNAME=***
export MLFLOW_TRACKING_PASSWORD=***
```

# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

  #with specific access

  1. EC2 access: It is a virtual machine
  2. ECR: Elastic Container registry to save your docker image in AWS

  #Description: About the deployment

  1. Build docker image of the source code
  2. Push your docker image to ECR
  3. Launch Your EC2 
  4. Pull Your image from ECR in EC2
  5. Launch your docker image in EC2

  #Policy:
  1. AmazonEC2ContainerRegistryFullAccess
  2. AmazonEC2FullAccess

## 3. Create ECR repo to store/save docker image
   - Save the URI: ***

## 4. Create EC2 machine (Ubuntu)

## 5. Open EC2 and Install docker in EC2 Machine:
   
   #optional
   sudo apt-get update -y
   sudo apt-get upgrade

   #required
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker ubuntu
   newgrp docker

## 6. Configure EC2 as self-hosted runner:
   Go to GitHub > Settings > Actions > Runners > New self-hosted runner > Choose OS > Run the commands one by one.

## 7. Setup GitHub secrets:
   AWS_ACCESS_KEY_ID=***
   AWS_SECRET_ACCESS_KEY=***
   AWS_REGION = us-east-1
   AWS_ECR_LOGIN_URI = 197137720867.dkr.ecr.ap-southeast-2.amazonaws.com
   ECR_REPOSITORY_NAME = simple-app
```
