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
## numpy.permutation 随机生成或者打乱顺序
np.random.permutation(10)
arr = np.arange(9).reshape((3, 3))
np.random.permutation(arr)
## sklearn train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)  random_state: 随机种子
# 如何划分train testset
## 随机hash划分sklearn train_test_split，当数量远远超过各种维度，split bias很小
## 按重要属性分层划分，train and test set都具主要属性代表性，当有及其重要属性，数据量不大
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
# 数据图形化，利用颜色和点大小表示各种维度，进行整体分析
## pandas plot
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
         s=housing["population"]/100, label="population",
         c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
) alpha：透明度，s：点的半径，label：右上角标签，c：颜色 predefined color map (option cmap) called jet 冷色系到暖色系
plt.legend()图例
# 皮尔逊系数只能查看线性相关性
corr_matrix = housing.corr(numeric_only=True)
## 相关性图形化
from pandas.plotting import scatter_matrix
scatter_matrix(housing[attributes], figsize=(12, 8))
