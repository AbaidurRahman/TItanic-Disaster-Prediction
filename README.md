This repository contains a comprehensive analysis and machine learning approach to predict survival on the Titanic, utilizing various models and techniques. The project is divided into multiple Jupyter notebooks, each focusing on different aspects of the machine learning pipeline, including data exploration, preprocessing, model training, evaluation, and deployment.

Project Structure
1. Survival on Titanic_ML.ipynb
This notebook is the core of the project, covering the entire machine learning workflow. It includes the following steps:

Data Loading and Exploration: Initial inspection of the Titanic dataset to understand the distribution and characteristics of features.
Data Preprocessing: Handling missing values, encoding categorical variables, and feature scaling.
Feature Selection: Selecting the most relevant features for model training.
Model Selection: Evaluating multiple machine learning models to choose the best-performing ones.
Model Training and Evaluation: Training models and assessing their performance on the validation set.
Hyperparameter Tuning: Fine-tuning model parameters using GridSearchCV for optimal performance.
Model Deployment: Finalizing the best model for deployment and making predictions on the test dataset.

2. main_ML.ipynb
This notebook replicates the entire process of Survival on Titanic_ML.ipynb, but with a modular approach. The models are trained and evaluated by calling functions from separate notebooks dedicated to each model. This setup enhances code reusability and organization.

3. logistic_regression.ipynb
This notebook is focused on the Logistic Regression model. It includes:

Data preprocessing steps specific to Logistic Regression.
Model training and evaluation.
Function definitions used in main_ML.ipynb.

4. random_forest.ipynb
This notebook is dedicated to the Random Forest model. It covers:

Data preprocessing tailored to Random Forest.
Model training and evaluation.
Function definitions used in main_ML.ipynb.

5. supportvm.ipynb
This notebook contains the Support Vector Machine (SVM) model, including:

SVM-specific data preprocessing.
Model training and evaluation.
Function definitions used in main_ML.ipynb.

6. Gradient_boosting.ipynb
This notebook is focused on the Gradient Boosting model, featuring:

Data preprocessing steps for Gradient Boosting.
Model training and evaluation.
Function definitions used in main_ML.ipynb.

7. gender_submission.csv
This file contains the final predictions on the test dataset, generated using the model with the best accuracy. The predictions are formatted as per Kaggle's submission requirements for the Titanic competition.

8. Hyperparameter Tuning with GridSearchCV
GridSearchCV was utilized across all models to fine-tune hyperparameters and optimize model performance. The results of this tuning process are integrated into the respective model notebooks.

Any queries related to the project and models can be asked through my email: 'abaidurrahman97@gmail.com'

This project is an ongoing active kaggle competition on supervised dataset containing features and labels.
