EXPERIMENT-2
POLYNOMIAL REGRESSION


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error , r2_score

url="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
df=pd.read_csv(url)
df.head()
x=df[['displacement']]
y=df['mpg']
print("Missing values in ",x.isnull().sum())
print("Missing values in mpg:",y.isnull().sum())

data=df[['displacement','mpg']].dropna()
x=data[['displacement']]
y=data['mpg']
print("Feature:")
print(x.head())
print("\nTarget:")
print(y.head())

plt.figure(figsize=(8,5))
plt.scatter(x,y)
plt.xlabel("Engine Displacement")
plt.ylabel("Miles Per Gallon (MPG)")
plt.grid(True)
plt.show()

x_train,x_test,y_train,y_test=train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)
print("Training samples:",len(x_train))
print("Testing samples:",len(x_test))

linear_model = LinearRegression()
linear_model.fit(x_train, y_train)
y_pred_linear = linear_model.predict(x_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)
print("Linear Regression Results")
print("-------------------------")
print("MSE:", mse_linear)
print("R-squared:", r2_linear)

results = []
polynomial_models = {}
results.append({
  'Model': 'Linear Regression',
  'Degree': 1,
  'MSE': mse_linear,
  'R-squared': r2_linear
})
for degree in range(2, 6):
    poly = PolynomialFeatures(degree=degree)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.transform(x_test)

    poly_model = LinearRegression()
    poly_model.fit(x_train_poly, y_train)
    y_pred_poly = poly_model.predict(x_test_poly)
    mse = mean_squared_error(y_test, y_pred_poly)
    r2 = r2_score(y_test, y_pred_poly)
    results.append({
        'Model': 'Polynomial Regression',
        'Degree': degree,
        'MSE': mse,
        'R-squared': r2
    })
    polynomial_models[degree] = {
        'poly': poly,
        'model': poly_model     }

results_df = pd.DataFrame(results)
results_df

x_plot = pd.DataFrame({
    'displacement': np.linspace(
        x['displacement'].min(), 
        x['displacement'].max(), 
        300) 
    })
plt.figure(figsize=(10, 6))

plt.scatter(
    x['displacement'],
    y,
    label='Actual Data'
)

y_plot_linear = linear_model.predict(x_plot)

plt.plot(
    x_plot['displacement'],
    y_plot_linear,
    label='Linear Regression'
)
for degree in range(2, 6):
   
 poly = polynomial_models[degree]['poly']
 poly_model = polynomial_models[degree]['model'] 
 x_plot_poly = poly.transform(x_plot)
 y_plot_poly = poly_model.predict(x_plot_poly)
 plt.plot(
        x_plot['displacement'],
        y_plot_poly,
        label=f'Polynomial Degree {degree}'
    )
plt.xlabel('Engine Displacement')
plt.ylabel('Miles Per Gallon (MPG)')
plt.title(
    'Linear and Polynomial Regression: Displacement vs MPG'
)
plt.legend()
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
bars = plt.bar(
    results_df['Degree'].astype(str),
    results_df['MSE']
)
plt.bar_label(bars, fmt='%.2f', padding=3)
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Comparison of MSE")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(
    results_df['Degree'],
    results_df['R-squared'],
    marker='o'
)
plt.xlabel("Polynomial Degree")
plt.ylabel("R-squared")
plt.title("Comparison of R-squared")
plt.grid(True)
plt.show()
