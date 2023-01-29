# Error Performance
## Root Mean Square Error (RMSE)
RMSE(X,h) = sqrt(Σ(h(xi) - yi)^2/m)
## Mean Absolute Error MAE also called Manhattan norm
MAE(X,h) = Σabs((h(xi) - yi)/m)
## RMSE compares 
The higher the norm index, the more it focuses on large values and neglects small ones. This is why the RMSE is more sensitive to outliers than the MAE. But when outliers are exponentially rare (like in a bell-shaped curve), the RMSE performs very well and is generally preferred.