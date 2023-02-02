# Error Performance
## Root Mean Square Error (RMSE)
RMSE(X,h) = sqrt(Σ(h(xi) - yi)^2/m)
## Mean Absolute Error MAE also called Manhattan norm
MAE(X,h) = Σabs((h(xi) - yi)/m)
## RMSE compares 
The higher the norm index, the more it focuses on large values and neglects small ones. This is why the RMSE is more sensitive to outliers than the MAE. But when outliers are exponentially rare (like in a bell-shaped curve), the RMSE performs very well and is generally preferred.
## tarfile的使用
tarfile.open(tgz_path)
# pandas
## dataframe 
info 查看概况每个字段类型 value_counts 查看单个分类字段各个类型上得总数 hist借用matlab包展示统计柱状图