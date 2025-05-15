# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import joblib

# Load the dataset
df = pd.read_csv('cars.csv')  # Assuming the data is saved as car_data.csv

# Data Exploration
print("=== Dataset Overview ===")
print(df.head())
print("\n=== Dataset Shape ===")
print(df.shape)
print("\n=== Dataset Columns ===")
print(df.columns)
print("\n=== Dataset Information ===")
print(df.info())
print("\n=== Descriptive Statistics ===")
print(df.describe())

# Check for missing values and duplicates
print(f"\n=== Missing Values:  {df.isnull().sum()}")
print(f"Number of duplicates: {df.duplicated().sum()}")

# ----------------Data Cleaning--------------
# Remove Bikes data and focus only on Cars
#--------------------------------------------
def clean_data(df):
    bike_keywords = ['Royal Enfield', 'KTM', 'Bajaj', 'Hero', 'Honda CB', 
                    'Yamaha', 'TVS', 'Activa', 'Splender', 'Pulsar']
    mask = df['Car_Name'].str.contains('|'.join(bike_keywords), case=False)
    df = df[~mask]
    
    # Remove impossible values
    df = df[(df['Driven_kms'] > 0) & 
            (df['Driven_kms'] < 300000) & 
            (df['Year'] > 2000) &
            (df['Selling_Price'] > 0.1)]
    
    return df


#-------------clean the data----------------
df_clean = clean_data(df)
# Drop unnecessary columns
# Feature engineering
df_clean['Vehicle_Age'] = 2025 - df_clean['Year']  # Assuming current year is 2023
#df_clean['Brand'] = df_clean['Car_Name'].str.split().str[0]
data = df_clean.drop(['Car_Name', 'Year'], axis=1)
print(data.head())


# Visualize the data
plt.figure(figsize=(15, 10))

# Distribution of Selling Price
plt.subplot(2, 2, 1)
sns.histplot(data['Selling_Price'], kde=True)
plt.title('Distribution of Selling Price')

# Relationship between features and selling price
plt.subplot(2, 2, 2)
sns.scatterplot(x='Present_Price', y='Selling_Price', data=data)
plt.title('Selling Price vs Present Price')

plt.subplot(2, 2, 3)
sns.scatterplot(x='Driven_kms', y='Selling_Price', data=data)
plt.title('Selling Price vs Kilometers Driven')

plt.subplot(2, 2, 4)
sns.boxplot(x='Fuel_Type', y='Selling_Price', data=data)
plt.title('Selling Price by Fuel Type')

plt.tight_layout()
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
#--------------copy the data to avoid modifying the original dataset--------------
corr_matrix = data.copy()
# Convert categorical columns to numerical using one-hot encoding
corr_matrix = pd.get_dummies(corr_matrix, columns=['Fuel_Type', 'Selling_type', 'Transmission'], drop_first=True)
corr_matrix = corr_matrix.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Car Price Dataset')
plt.show()

# Prepare data for modeling
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps
numeric_features = ['Present_Price', 'Driven_kms', 'Vehicle_Age']
categorical_features = ['Fuel_Type', 'Selling_type', 'Owner', 'Transmission']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)])

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
 }

# Train and evaluate models
results = {}
for name, model in models.items():
    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'RMSE': rmse,
        'R2 Score': r2,
        'Model': pipeline
    }
    
    print(f"\n=== {name} ===")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")

# Find the best model
best_model_name = min(results, key=lambda x: results[x]['RMSE'])
best_model = results[best_model_name]['Model']
print(f"\nBest Model: {best_model_name} with RMSE: {results[best_model_name]['RMSE']:.2f}")

# Feature Importance for tree-based models
if hasattr(best_model.named_steps['model'], 'feature_importances_'):
    # Get feature names after one-hot encoding
    ohe_features = best_model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features)
    all_features = numeric_features + list(ohe_features)
    
    importances = best_model.named_steps['model'].feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title(f'Feature Importance ({best_model_name})')
    plt.tight_layout()
    plt.show()


def predict_car_price(Present_price, driven_kms, fuel_type, selling_type, Transmission,Owner, vehicle_age):
    """Predict car price based on input features."""
    # Create DataFrame with correct column names and order
    input_data = pd.DataFrame({
        'Present_Price': [Present_price],
        'Driven_kms': [driven_kms],  # Changed to match training data
        'Fuel_Type': [fuel_type],
        'Selling_type': [selling_type],  # Changed to match training data
        'Transmission': [Transmission],
        'Owner': [Owner],
        'Vehicle_Age': [vehicle_age]
    })
    
    # Ensure categorical columns match training data categories
    categorical_cols = ['Fuel_Type', 'Selling_type', 'Transmission', 'Owner']
    for col in categorical_cols:
        input_data[col] = input_data[col].astype('category')
    
    # Make prediction
    return best_model.predict(input_data)[0]

# Example usage
try:
    example_pred = predict_car_price(
        Present_price=5.59,
        driven_kms=27000,
        fuel_type='Petrol',
        selling_type='Dealer',
        Transmission='Manual',
        Owner=0,
        vehicle_age=2
       )
    print(f"\nPredicted Selling Price: ₹{example_pred:.2f}")  # Assuming INR currency
except Exception as e:
    print(f"Prediction failed: {str(e)}")
# Compare with actual
example_actual = data.iloc[0]['Selling_Price']
print(f"Actual Selling Price: ${example_actual:.2f}")
print(f"Difference: ${(example_pred - example_actual):.2f}")

# Save the best model
joblib.dump(best_model, 'car_price_model.pkl')
print("\nModel saved as 'car_price_model.pkl'")