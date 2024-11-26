import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('data/gemeenten.csv', encoding='ISO-8859-1', sep=';')

# Some parsing from the raw sheet for python
# Delete percentage signs and comma seperators to dots
data['Alles elektrisch'] = data['Alles elektrisch'].str.replace('%', '')
data['Stemmen op PVV'] = data['Stemmen op PVV'].str.replace('%', '').astype(float) / 100 
data['Stemmen op coalitie'] = data['Stemmen op coalitie'].str.replace('%', '').astype(float) / 100 
data['Alles elektrisch'] = data['Alles elektrisch'].str.replace(',', '.').astype(float)
data['Koopwoningen'] = data['Koopwoningen'].str.replace(',', '.').astype(float)

# Feature Selection and Scaling (same as before)
features = ['Stemmen op PVV', 'Stemmen op coalitie', 'WOZ', 'Bevolkingsdichtheid', 'Koopwoningen']
target = 'Alles elektrisch'

scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# Split data (same as before)
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100,  # Number of trees
                             max_depth=None,  # Or specify a max depth
                             random_state=42)
model.fit(X_train, y_train)

# Evaluate model using cross-validation (same as before)
cv_scores = cross_val_score(model, data[features], data[target], cv=5)

# Make predictions (same as before)
y_pred = model.predict(X_test)

# Evaluate the model (same metrics as before)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
adjusted_r2 = 1 - (1-r2)*(len(y_test)-1)/(len(y_test)-len(features)-1)

print(f'Mean Absolute Error: {mae:.2f}')
print(f'Mean Squared Error: {mse:.2f}')
print(f'R² Score: {r2:.2f}')
print(f'Adjusted R² Score: {adjusted_r2:.2f}')
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean cross-validation score: {np.mean(cv_scores):.2f}')

# Feature importance (Random Forest's feature importance)
feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print('\nFeature Importance:')
print(feature_importances)

# ... (Predictions for all neighbourhoods remain the same) ...