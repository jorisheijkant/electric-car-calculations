import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data/gemeenten_2.csv', encoding='ISO-8859-1', sep=';')

# Some parsing from the raw sheet for python
# Delete percentage signs and comma seperators to dots
data['Elektrisch per inwoner'] = data['AANTAL VOERTUIGEN'].astype(int) / data['Inwoners'].replace(',', '.').astype(float)
data['Stemmen op PVV'] = data['Stemmen op PVV'].str.replace('%', '').astype(float) / 100 
data['Stemmen op coalitie'] = data['Stemmen op coalitie'].str.replace('%', '').astype(float) / 100 
data['Elektrisch percentage (oud)'] = data['Alles elektrisch'].str.replace('%', '')
data['Elektrisch percentage (oud)'] = data['Elektrisch percentage (oud)'].str.replace(',', '.').astype(float)
data['Koopwoningen'] = data['Koopwoningen'].str.replace(',', '.').astype(float)

data = data.dropna() 

correlation = data[['Elektrisch per inwoner', 'WOZ', 'Stemmen op coalitie', 'Bevolkingsdichtheid']].corr(method='pearson')
print(correlation)

data.to_csv("calculations.csv")

sns.regplot(x='Bevolkingsdichtheid', y='Elektrisch per inwoner', data=data, ci=None, line_kws={"color": "red"})
plt.title('Electric cars vs population density')
plt.xlabel('Average population density (people per km2)')
plt.ylabel('Electric cars per inhabitant')
plt.show()

