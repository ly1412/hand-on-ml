# 1. pca一般是数据准备，利用下游算法结果
# 2. minimize reconstruction error
# 3. rbf高斯核，会将数据映射到无限维，无法直接计算，对无限维数据降维后，利用近似点，进行reconstruction，从而计算近似reconstruction error，可利用回归算法计算