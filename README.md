# ML_Project40-PredictingAdmissionIntoUCLA

#### UCLA Admission Prediction

This project explores predicting the probability of admission for applicants to UCLA using a machine learning model. The model is trained on a dataset containing features such as GRE scores, TOEFL scores, university ratings, and past research experience.

#### Data

The project utilizes a dataset named admission_predict.csv. This dataset is expected to have the following columns:
GRE Score: Applicant's GRE score

TOEFL Score: Applicant's TOEFL score

University Rating: Rating of the applicant's university on a scale

SOP: Statement of Purpose score (float)

LOR: Letter of Recommendation score (float)

CGPA: Applicant's CGPA score

Research: Indicates if the applicant has research experience (0 or 1)

Probability: Admission probability (between 0 and 1)


### Code Structure

### Exploration and Cleaning:

Loads the CSV data into a pandas DataFrame.

Performs basic data exploration and cleaning tasks.

Handles missing values by replacing them with NaN.

## Visualization:
Creates histograms to visualize the distribution of features.

### Model Building:

Splits the data into features (X) and labels (y).

Implements a function find_best_model to perform GridSearchCV with various machine learning algorithms to identify the best model for the problem.

Based on the GridSearch results, selects Linear Regression as the model due to its highest accuracy.

Evaluates the model's performance using cross-validation with cross_val_score.

Splits the data into training and testing sets using train_test_split.

Trains the Linear Regression model with normalization.

Evaluates the model's performance on the testing set using model.score.


### Prediction:
Demonstrates how to use the trained model to predict the admission probability for new applicants based on their features.


## How to Run
Ensure you have the necessary libraries installed (pandas, numpy, matplotlib, scikit-learn).

Save the code as a Python file (e.g., admission_prediction.py).

Execute the script from your terminal: python admission_prediction.py

This will run the code, performing data exploration, model training, and displaying sample predictions.

### Disclaimer

This is a sample project for educational purposes. The accuracy of the model may not reflect real-world UCLA admission decisions, which consider a multitude of factors beyond the scope of this dataset.
