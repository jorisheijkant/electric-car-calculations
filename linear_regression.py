import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv('data/gemeenten.csv', encoding='ISO-8859-1', sep=';')

# Some parsing from the raw sheet for python
# Delete percentage signs and comma seperators to dots
data['Alles elektrisch'] = data['Alles elektrisch'].str.replace('%', '')
data['Stemmen op PVV'] = data['Stemmen op PVV'].str.replace('%', '').astype(float) / 100 
data['Stemmen op coalitie'] = data['Stemmen op coalitie'].str.replace('%', '').astype(float) / 100 
data['Alles elektrisch'] = data['Alles elektrisch'].str.replace(',', '.').astype(float)
data['Koopwoningen'] = data['Koopwoningen'].str.replace(',', '.').astype(float)

parameters = ['Stemmen op PVV', 'Stemmen op coalitie', 'WOZ', 'Bevolkingsdichtheid', 'Koopwoningen']

X = data[parameters]  
y = data['Alles elektrisch']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Absolute Error: {mae:.2f}')
print(f'Mean Squared Error: {mse:.2f}')
print(f'R² Score: {r2:.2f}')

all_neighbourhoods = data.copy()  
new_predictions = model.predict(all_neighbourhoods[parameters])
all_neighbourhoods['predicted_electric'] = new_predictions

feature_importances = pd.Series(model.coef_, index=X.columns).sort_values(ascending=False)
print('Feature Importance:')
print(feature_importances)