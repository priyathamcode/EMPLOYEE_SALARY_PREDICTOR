# Income Prediction Streamlit App 💰

This project is a machine learning web application built with Streamlit that predicts whether an individual's annual income is likely to be more or less than $50,000. It uses a Logistic Regression model trained on the well-known UCI Adult Dataset.

The app provides an interactive interface for users to input demographic and employment information and receive a real-time prediction.

-----

## Features

  - Interactive UI: A user-friendly form in the sidebar to input feature data.
  - Real-Time Predictions: Instantly classifies the input data into `>50K` or `<=50K` income brackets.
  - Prediction Confidence: Displays the model's confidence score for each prediction.
  - Model Transparency: Shows key performance metrics (Accuracy, Precision, Recall, F1-Score) of the trained model.
  - Data Explorer: Includes an expandable section to view and filter the training dataset.

-----

## Tech Stack

  - Language: Python 3
  - Framework: Streamlit
  - Libraries:
      - Pandas
      - Scikit-learn
      - NumPy

-----

## How to Run

To run this project locally, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. Create a `requirements.txt` file with the following content:

```
streamlit
pandas
scikit-learn
numpy
```

3. Install the required libraries:

```bash
pip install -r requirements.txt
```

4. Get the dataset:
Download the `adult 3.csv` dataset and place it in the root directory of the project.

5. Run the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your default web browser.

-----

## Dataset

This model is trained on the UCI Adult Dataset, which contains features like age, workclass, and education to predict whether an income exceeds $50K/yr.
