import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
predicted = body_reg.predict(pd.DataFrame([[user_input]], columns=['Brain']))

# Read data
dataframe = pd.read_fwf('brain_body.txt')
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]

# Train model
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

# Print equation
print("Regression Equation: y =", round(body_reg.coef_[0][0], 2), "* x +", round(body_reg.intercept_[0], 2))

# Predict
brain_input = float(input("Enter brain weight to predict body weight: "))
predicted = body_reg.predict(pd.DataFrame([[user_input]], columns=['Brain']))
print("Predicted body weight:", round(predicted_body[0][0], 2))

# Plot
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()
