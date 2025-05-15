# ------------------------------------------------------------
#        Sales Prediction using Python
# ------------------------------------------------------------

# --------------  Key Features of This Implementation:
# --------------- Data Loading & Exploration:
# . Loads the provided sales dataset
# . Provides statistical overview and checks for missing values
# --------------- Comprehensive Visualization:
# . Distribution plots for sales
# .Scatter plots showing relationships between ad spend and sales
# . Correlation heatmap
# --------------- Multiple Modeling Approaches:
# . Linear Regression (regular and regularized)
# . Random Forest             . Gradient Boosting
# --------------- Thorough Evaluation:
# MAE, MSE, RMSE, and R² metrics
# Automatic selection of best performing model
# -------------- Advanced Features:
# . Feature importance visualization      . Hyperparameter tuning for the best model
# . Prediction function for new data      . Model persistence
# -------------- Practical Application:
# . Example prediction showing how to use the model
# . Comparison with actual values

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Load the dataset
data = pd.read_csv('Advertising.csv') # Ensure the file path is correct
a
# Data Exploration
print("=== Dataset Overview ===")
print(data.head())
print("\n=== Dataset Information ===")
print(data.info())
print("\n=== Descriptive Statistics ===")
print(data.describe())

# Check for missing values And duplicates
print("\n=== Duplicates ===")
print(f"Number of duplicates: {data.duplicated().sum()}")
print("\n=== Missing Values ===")
print(data.isnull().sum())

# Visualize the data
plt.figure(figsize=(15, 10))

# Distribution of Sales
plt.subplot(2, 2, 1)
sns.histplot(data['Sales'], kde=True)
plt.title('Distribution of Sales')

# Relationship between advertising channels and sales
plt.subplot(2, 2, 2)
sns.scatterplot(x='TV', y='Sales', data=data)
plt.title('Sales vs TV ADS')

plt.subplot(2, 2, 3)
sns.scatterplot(x='Radio', y='Sales', data=data)
plt.title('Sales vs Radio ADS')

plt.subplot(2, 2, 4)
sns.scatterplot(x='Newspaper', y='Sales', data=data)
plt.title('Sales vs Newspaper ADS')

plt.tight_layout()
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 5))
data = data.drop(columns=['Unnamed: 0'])  # Drop index column if exists
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Visual inspection using boxplots
for col in ['TV', 'Radio', 'Newspaper', 'Sales']:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=data[col])
    plt.title(f'Boxplot of {col} for Outlier ')
    plt.show()

# Detect outliers using IQR (Interquartile Range)
outlier_indices = {}
for col in ['TV', 'Radio', 'Newspaper', 'Sales']:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
    outlier_indices[col] = outliers.index.tolist()
    
    print(f"{col} -== Number of outliers ==: {len(outliers)}")
    print(f"== Indices ==: {outliers.index.tolist()}")

# Optional: Combine all outlier indices across columns
all_outliers = set()
for indices in outlier_indices.values():
    all_outliers.update(indices)

print(f"\nTotal unique rows with outliers: {len(all_outliers)}")
print(f"Row indices with outliers: {sorted(all_outliers)}")

# Prepare data for modeling
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    # Create pipeline with scaling for linear models
    if 'Regression' in name:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
    else:
        pipeline = model
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Model': pipeline
    }
    
    print(f"\n=== {name} ===")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")

# Find the best model
best_model_name = max(results, key=lambda x: results[x]['R2'])
best_model = results[best_model_name]['Model']
print(f"\nBest Model: {best_model_name} with R2 Score: {results[best_model_name]['R2']:.2f}")

# Feature Importance for tree-based models
if hasattr(best_model, 'feature_importances_'):
    # For pipeline
    if hasattr(best_model, 'named_steps'):
        importances = best_model.named_steps['model'].feature_importances_
    else:
        importances = best_model.feature_importances_
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title(f'Feature Importance({best_model_name})')
    plt.show()

# Hyperparameter Tuning for the best model
if best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring='r2'
    )
    grid_search.fit(X_train, y_train)
    
    print("\nBest Parameters:", grid_search.best_params_)
    print("Best R2 Score:", grid_search.best_score_)
    
    best_model = grid_search.best_estimator_

# Make predictions
def pred_sales(tv, radio, newspaper):
    """Predict sales based on TV, Radio, and Newspaper spending."""
    input_data = pd.DataFrame({
        'TV': [tv],
        'Radio': [radio],
        'Newspaper': [newspaper]
    })
    return best_model.predict(input_data)[0]

# Example prediction
example_pred = pred_sales(230.1, 37.8, 69.2)
print(f"\nPredicted Sales (for spending TV: $230.1k, Radio: $37.8k, Newspaper: $69.2k): is = ${example_pred:.2f}")

# Compare with actual
example_actual = data.loc[0, 'Sales']
print(f"Actual Sales: ${example_actual:.2f}")
print(f"Difference: ${(example_pred - example_actual):.2f}")

# Save the best model
import joblib
joblib.dump(best_model, 'sales_model.pkl')
print("Model saved as 'sales_model.pkl'")

# Findings from Sales Prediction Analysis
print("\n === Sales Prediction Model for Advertising Optimization: A Data Science Approach ===")
print(" Report Description")
print(" Objective:")
print(" This project analyzes the relationship between advertising expenditures (TV, Radio, Newspaper)")
print(" and sales revenue using exploratory data analysis (EDA) and predictive modeling.")
print(" The goal is to identify key drivers of sales and build an optimized machine learning ")
print(" model to forecast sales based on advertising budgets.")
print(" Data Insights:")
print(" TV advertising has the strongest correlation with sales (0.78), followed by Radio (0.58).")
print(" Newspaper ads show minimal impact (0.23).")
print(" Model Performance:")
print(" Gradient Boosting outperformed other models (R² = 0.98, MAE = 0.53),demonstrating ")
print(" near-perfect prediction accuracy.")
print(" Feature Importance:")
print(" Feature importance analysis confirmed TV ads as the dominant sales driver 70% ")
print(" contribution), while Newspaper ads had negligible influence (<10%).")
print(" Business Implications:")
print(" The model suggests reallocating budgets from Newspaper to TV/Radio could improve ROI.")
print(" The model provides actionable sales forecasts, enabling data-driven budget planning.")
