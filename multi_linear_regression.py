import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import warnings


warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

poly_features = PolynomialFeatures(degree=2, include_bias=False)

lin_reg = LinearRegression()
reg = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('linear', lin_reg)
])
reg.fit(X, y)

print(lin_reg.intercept_, lin_reg.coef_)

X_new = np.linspace(-3, 3, num=10).reshape((10, 1))
y_predict = reg.predict(X_new)
plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.show()
