import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data/gemeenten.csv', encoding='ISO-8859-1', sep=';')

# Some parsing from the raw sheet for python
# Delete percentage signs and comma seperators to dots
data['Alles elektrisch'] = data['Alles elektrisch'].str.replace('%', '')
data['Stemmen op PVV'] = data['Stemmen op PVV'].str.replace('%', '').astype(float) / 100 
data['Stemmen op coalitie'] = data['Stemmen op coalitie'].str.replace('%', '').astype(float) / 100 
data['Alles elektrisch'] = data['Alles elektrisch'].str.replace(',', '.').astype(float)
data['Koopwoningen'] = data['Koopwoningen'].str.replace(',', '.').astype(float)

# Check for missing or invalid data
print(data.isnull().sum())
data = data.dropna() 

correlation = data[['Alles elektrisch', 'WOZ', 'Stemmen op coalitie', 'Bevolkingsdichtheid']].corr(method='pearson')
print(correlation)

sns.regplot(x='Bevolkingsdichtheid', y='Alles elektrisch', data=data, ci=None, line_kws={"color": "red"})
plt.title('Electric cars vs population density')
plt.xlabel('Average population density (people per km2)')
plt.ylabel('Percentage of electric cars')
plt.show()
