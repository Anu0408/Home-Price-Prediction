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

Clone the repository

```bash
https://github.com/sekhar4ml/Home-Price-Prediction.git
```

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