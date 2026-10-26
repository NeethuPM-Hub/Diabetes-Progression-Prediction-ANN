# Diabetes Disease Progression Prediction Using ANN

## 📌 Project Overview

This project focuses on predicting diabetes disease progression using an Artificial Neural Network (ANN) and the Diabetes dataset available in Scikit-learn.

The objective is to build a regression model, evaluate its performance, and experiment with different neural network architectures and regularization techniques to understand their impact on prediction accuracy.

**Problem Type:** Regression  
**Best Performing Model:** Baseline ANN  
**Framework:** TensorFlow/Keras  
**Environment:** Google Colab / Jupyter Notebook

## 🎯 Objectives

- Load and preprocess the Scikit-learn Diabetes dataset.
- Perform Exploratory Data Analysis (EDA).
- Normalize input features for ANN training.
- Build and train ANN regression models.
- Evaluate model performance using MSE, RMSE, MAE, and R² score.
- Experiment with different architectures and L2 regularization.
- Compare model performance and identify the best-performing configuration.

## 📂 Dataset Description

The dataset is loaded using `sklearn.datasets.load_diabetes()`.

- **Number of samples:** 442
- **Number of independent features:** 10
- **Target variable:** Quantitative measure of diabetes disease progression

The features include age, sex, body mass index (BMI), average blood pressure, and six blood serum measurements.

The target represents a quantitative measure of disease progression one year after baseline.

**Note:** This is a regression problem, not a diabetes diagnosis or classification task.

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow/Keras
- Google Colab

## ⚙️ Project Workflow

### 1. Data Loading and Preprocessing

- Loaded the Diabetes dataset using Scikit-learn.
- Checked for missing values and duplicate records.
- Separated the independent features and target variable.
- Split the dataset into training and testing sets using an 80:20 ratio.
- Applied `StandardScaler` to normalize the input features.
- Fitted the scaler only on the training data to avoid data leakage.

### 2. Exploratory Data Analysis (EDA)

EDA was performed to understand the distributions and relationships within the dataset.

Visualizations included:

- Target variable distribution.
- Distributions of independent features.
- Correlation heatmap.
- Scatter plots showing relationships between selected features and the target.

### 3. ANN Model Development

Four ANN configurations were developed and evaluated.

| Model | Architecture / Technique |
|---|---|
| Baseline ANN | One hidden layer with 32 neurons |
| Improved ANN | Two hidden layers with 64 and 32 neurons |
| Smaller ANN | One hidden layer with 16 neurons |
| L2 Regularized ANN | One hidden layer with 32 neurons and L2 regularization |

The models used:

- ReLU activation in hidden layers.
- Linear activation in the output layer.
- Adam optimizer.
- Mean Squared Error (MSE) loss function.
- Early stopping to help control overfitting.

### 4. Model Evaluation

The models were evaluated using the following regression metrics:

- **Mean Squared Error (MSE):** Measures the average squared prediction error.
- **Root Mean Squared Error (RMSE):** Measures prediction error on the target's scale.
- **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted values.
- **R² Score:** Measures the proportion of variation in the target explained by the model relative to a mean-prediction baseline.

## 📊 Results and Model Comparison

The following results were obtained during the final reported experiment.

| Model | MSE ↓ | RMSE ↓ | MAE ↓ | R² ↑ |
|---|---:|---:|---:|---:|
| **Baseline ANN** | **2958.297** | **54.390** | **42.854** | **0.442** |
| Improved ANN | 2986.251 | 54.647 | 43.589 | 0.436 |
| Smaller ANN (16 neurons) | 3441.796 | 58.667 | 46.182 | 0.350 |
| L2 Regularized ANN | 2970.503 | 54.502 | 43.398 | 0.439 |

*Lower MSE, RMSE, and MAE indicate better performance. Higher R² indicates a better fit relative to the mean-prediction baseline.*

### 🏆 Best Performing Model: Baseline ANN

The Baseline ANN achieved the best performance among the four configurations in this experiment.

- **MSE:** 2958.297
- **RMSE:** 54.390
- **MAE:** 42.854
- **R² Score:** 0.442

The model explains approximately 44.2% of the variation in the test-set target relative to predicting its mean.

## 🔍 Key Findings

- The Baseline ANN outperformed the other three configurations across all four evaluation metrics.
- Adding another hidden layer did not improve predictive performance in this experiment.
- The smaller ANN with 16 neurons produced the highest prediction errors.
- L2 regularization produced results close to the baseline but did not improve its test metrics.
- Increasing model complexity does not necessarily lead to better generalization.
- The results demonstrate the importance of evaluating different architectures instead of assuming that a more complex model will perform better.

## 🚀 How to Run the Project

1. Clone or download this repository.
2. Open the Jupyter Notebook in Google Colab or a local Jupyter environment.
3. Install the required libraries if they are not already available:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
   ```

4. Run the notebook cells sequentially.
5. Review the preprocessing steps, EDA visualizations, model training, evaluation metrics, and comparison results.

The dataset is loaded directly from Scikit-learn, so no separate dataset download is required.

## 📁 Repository Structure

```text
Diabetes-ANN-Regression/
├── Diabetes_ANN_Deep_Learning_Assignment.ipynb
└── README.md
```

## ✅ Conclusion

This project demonstrates the application of Artificial Neural Networks to a regression problem using the Scikit-learn Diabetes dataset.

Four ANN configurations were evaluated, and the Baseline ANN achieved the best test performance in the reported experiment, with an MSE of 2958.297 and an R² score of 0.442.

The experiments highlight that model complexity and regularization do not automatically improve prediction accuracy. Further hyperparameter tuning and cross-validation could help evaluate the stability of these findings and identify opportunities for improvement.

**Disclaimer:** This project is intended for educational purposes only. The model is not intended for clinical diagnosis, medical decision-making, or treatment recommendations.

## 👩‍💻 Author

Developed as part of a Deep Learning Module-End Assignment.
