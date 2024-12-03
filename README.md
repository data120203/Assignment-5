# Assignment-5
Neural Networks algorithm for drug classification

## Project Description

This project aims to classify patients into appropriate drug categories based on demographic and clinical measurements. The dataset, drugdataset.csv, contains information on patient characteristics and prescribed drug types. We use a Neural Network model to predict drug type based on input variables such as Age, Sex, Blood Pressure, Cholesterol, and Sodium-to-Potassium ratio.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

You will need the following libraries installed in your Python environment:

pandas
numpy
matplotlib
scikit-learn
You can install the required libraries using pip:

```
pip install pandas numpy matplotlib scikit-learn
```

### Installing

1. Clone the repository or download the project files.
2. Place the drugdataset.csv file in the project directory.
3. Run the provided Python script to preprocess data, train the Neural Network, and evaluate its performance.

## Running the tests

1. Load the dataset and inspect it for completeness.
2. Encode categorical variables.
3. Standardize numerical features using StandardScaler.
4. Split the data into training and testing sets.
5. Train a Neural Network model using MLPClassifier.
5. Evaluate the model.
6. Analyze the classification report for precision, recall, and F1-score.

```
# Load Dataset
drugdataset = pd.read_csv('./drugdataset.csv')

# Preprocess and train Neural Network
x_train2 = sc.fit_transform(x_train)
mlp.fit(x_train2, y_train)
predictions = mlp.predict(x_test2)

# Evaluate performance
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions, target_names=target_names))
```

### Break down into end to end tests

The script includes automated evaluation to test the accuracy of predictions:

1. Confusion Matrix: Shows the distribution of predicted vs. actual labels.
2. Classification Report: Includes metrics like precision, recall, and F1-score.

```
Confusion Matrix:
[[ 5  0  0  0  0]
 [ 0  2  0  0  1]
 [ 0  0  3  0  0]
 [ 0  0  0 11  0]
 [ 0  0  0  1 17]]

Classification Report:
              precision    recall  f1-score   support
       drugA       1.00      1.00      1.00         5
       drugB       1.00      0.67      0.80         3
       drugC       1.00      1.00      1.00         3
       drugX       0.92      1.00      0.96        11
       drugY       0.94      0.94      0.94        18
```


## Deployment

To deploy this system for real-world use:

1. Prepare a larger dataset for training to improve generalization.
2. Set up an API for real-time drug predictions based on input features.

## Built With

* Python - Programming Language
* scikit-learn - Machine Learning Library
* pandas - Data Analysis and Manipulation
* matplotlib - Visualization

## Authors

* **Student 100941875** - *Initial work*

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* John Hughes for project inspiration.
