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
## drop函数
去掉一列：set.drop(["income_cat"], axis=1, inplace=True)
# numpy
## numpy.permutation 随机生成或者打乱顺序
np.random.permutation(10)
arr = np.arange(9).reshape((3, 3))
np.random.permutation(arr)
## numpy.reshape(-1, 1)
第一个参数表示增加维度，第二个为每个维度元素，也就是len()对第二个参数必须整除
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
# 相关性计算（皮尔逊系数只能查看线性相关性）
corr_matrix = housing.corr(numeric_only=True)
## 相关性图形化
from pandas.plotting import scatter_matrix
scatter_matrix(housing[attributes], figsize=(12, 8))
# 利用当前特征产生新特征，并计算相关程度
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
# 数据清洗
## 删除空数据
housing.dropna(subset=["total_bedrooms"])
## 删除整个属性
housing.drop("total_bedrooms", axis=1)
## 以均值或者中值等默认值填充
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median)
计算出的默认值，可以继续填充test data以及new data
# Imputer 使用
## imputer构建
from sklearn.impute import SimpleImputer
## 设置策略
imputer = SimpleImputer(strategy='mean')
## 删除非数字列
housing_num = housing.drop('ocean_proximity', axis=1)
## 训练数据集
imputer.fit(housing_num)
print(imputer.statistics_)
print(housing_num.mean().values)
## 转化矩阵
X = imputer.transform(housing_num)   X 是一个稀疏矩阵或者多维numpy array
housing_pandas = pd.DataFrame(X, columns=housing_num.columns)   再次转回pandas
# Scikit-Learn Design
## 三大接口一致性 Estimators -> fit()、Transformers -> transform()、Predictors -> predict()
## 所有超参均可直接访问estimator.name，Estimators的已学的参数也可直接访问estimator.name_
## 公用返回：NumPy arrays or SciPy sparse matrices，简单的超参：Python strings or numbers
## 合理的默认值
# 文本&分类属性
## 分类编码
将categorical属性编码成数字
## LabelEncoder用法
### 构建
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
### 找出对应列
housing_cat = housing['ocean_proximity']
### 训练转化
housing_cat_encoded = encoder.fit_transform(housing_cat)
## onehotencoder
### 构建
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
### 训练
housing_cat_onehot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
### 返回是sparse matrix 使用toarray方法变回numpy array
## LabelBinarizer = LabelEncoder + OneHotEncoder
### 构建
from sklearn.preprocessing import LabelBinarizer
### 指定输出sparse matrix 默认是多维数组
encoder = LabelBinarizer(sparse_output=True)
housing_cat_onehot = encoder.fit_transform(housing_cat)


