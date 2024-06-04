import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
import seaborn as sns
import tensorflow as tf


def sigma(xn):
    # print(xn)
    n = len(xn)
    # print(n)
    sum_sq = xn.sum() ** 2
    sq_sum = 0
    for x in xn:
        sq_sum += x ** 2
    return np.sqrt((sq_sum - (sum_sq / n)) / (n - 1))


def preprocess(filepath: str, filename: str, max_miss=10):
    data = pd.read_csv(filepath + filename, encoding='GBK')
    data.drop(['时间'], axis=1, inplace=True)
    variable_labels = data.keys()
    variable_value_range = pd.read_excel(filepath + '附件四：354个操作变量信息.xlsx', )

    for k in variable_labels:
        miss_part = data[k][data[k] == 0]
        length = len(miss_part)
        if length >= max_miss:  # case 1 and 2
            # print(k)
            data.drop([k], axis=1, inplace=True)
        elif 1 <= length < max_miss:  # case 3
            data.loc[miss_part.index, k] = data[k].mean()

    variable_labels = data.keys()
    for k in variable_labels:  # case 4
        res = ['', '']
        i = 0
        # print(variable_value_range[variable_value_range['位号'] == k]['取值范围'].values[0])
        for c in variable_value_range[variable_value_range['位号'] == k]['取值范围'].values[0]:
            if res[i] == '' and c == '-':
                res[i] += c
            elif c.isdigit():
                res[i] += c
            elif c == '.':
                res[i] += c
            elif c == '-':
                i += 1
            else:
                continue
        _min, _max = float(res[0]), float(res[1])
        # print(_min, _max)
        # print(find_out_of_bounding.index)
        if data[k].min() < _min or data[k].max() > _max:
            data.drop([k], axis=1, inplace=True)
        #     print(data[k])
        #     data.drop(data.index[find_out_of_bounding.index], inplace=True)
        # break

    variable_labels = data.keys()
    for k in variable_labels:
        xn = data.loc[:, k]
        si = sigma(xn)
        for _, x in enumerate(xn):
            v = np.abs(x - xn.mean())
            if v > 3 * si:
                data.drop(data.index[_], inplace=True)
    return data


def plot(mat):
    sns.heatmap(mat.corr(), annot=True)
    # sns.pairplot(mat)
    plt.show()


def plot_train_loss(loss, title):
    plt.plot(loss)
    plt.title('Model %s loss' % title)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


def plot_predicted_result(y_true, y_pred, time_step):
    fig, ax = plt.subplots()
    ax.plot(y_true, label='True')
    ax.plot(y_pred, label='Predicted')
    plt.xticks(list(range(len(time_step))), time_step, rotation=70)
    ax.set_ylabel('RON Loss')
    ax.set_xlabel('Time step')
    ax.legend()
    plt.title('Compared')
    plt.show()


path = '数模题/'
data = pd.read_excel(path + '附件一：325个样本数据.xlsx', header=1)
data1 = preprocess(path, '285号原始样本.csv')
data2 = preprocess(path, '313号原始样本.csv')

data1 = data1.mean()
data2 = data2.mean()

n285_sample = pd.DataFrame({'285号原始样本': data.loc[285, data1.index], '285号样本': data1}, index=data1.index)
n285_sample.to_csv(path + '285号样本对比.csv', index=False)

n313_sample = pd.DataFrame({'313号原始样本': data.loc[313, data2.index], '313号样本': data2})
n313_sample.to_csv(path + '313号样本对比.csv', index=False)

print('285号样本变化变量对比')
for labels, i, j in zip(data1.index, data.loc[285, data1.index], data1):
    if i != j:
        print(labels, i, j)

print('313号样本变化变量对比')
for labels, i, j in zip(data2.index, data.loc[313, data2.index], data2):
    if i != j:
        print(labels, i, j)

data.loc[285, data1.index] = data1
data.loc[313, data2.index] = data2

data.to_excel(path + 'res.xlsx', index=False)

ron_loss = data.loc[1:, 'Unnamed: 11'].values.astype(np.float32)  # 目标变量
ron_loss = ron_loss[::-1]
print(ron_loss)

time_step = pd.to_datetime(data.loc[1:, '时间']).dt.strftime('%Y/%m/%d')
time_step = time_step.iloc[::-1]
print(time_step)

data.drop(['样本编号', '时间', 'Unnamed: 11'], axis=1, inplace=True)
data = data.loc[1:, :]
data = data.iloc[::-1]

select_features = list(data.keys()[:13]) + [
    'S-ZORB.FT_1001.PV',
    'S-ZORB.FT_1004.PV',
    'S-ZORB.TE_1001.PV',
    'S-ZORB.TE_1201.PV',
    'S-ZORB.FC_1203.PV',
    'S-ZORB.FT_5201.PV',
    'S-ZORB.TE_1101.DACA',
    'S-ZORB.PT_7107B.DACA',
    'S-ZORB.PT_7103B.DACA',
    'S-ZORB.TE_5008.DACA',
    'S-ZORB.TE_7108B.DACA',
    'S-ZORB.TE_7106B.DACA',
    'S-ZORB.AT_1001.DACA',
    'S-ZORB.TE_5002.DACA',
    'S-ZORB.TE_1105.PV',
    'S-ZORB.FT_5104.PV']
data = data[select_features]
# print(data.corr())
# plot(data)  # 绘制数据相关系数图

train = zscore(np.array(data.values, dtype=np.float32), axis=1)  # 数据中心化

# pca = PCA(30)  # PCA 模型
# train = pca.fit_transform(train,)  # 降维
#
# exp_var = pd.DataFrame(pca.explained_variance_)  # 解释方差
# comp = pd.DataFrame(pca.components_)  # 成分矩阵
#
# exp_var.to_csv(path + 'explained_variance.csv', index=False)
# comp.to_csv(path + 'components.csv', index=False)
#
# dem_data = pd.DataFrame(train)
# dem_data.to_csv(path + 'dem.csv', index=False)  # 主成分数据

# scaler = MinMaxScaler()  # 归一化
scaler = StandardScaler()  # 标准化
train = scaler.fit_transform(train)  # 数据归一化

train_X, test_X, train_y, test_y = train_test_split(train, ron_loss, test_size=0.1, random_state=0)  # 数据集划分
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# 建立 LSTM 模型
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])),
    tf.keras.layers.Dropout(0.2),  # 防止过拟合
    tf.keras.layers.Dense(1)  # 输出形状为一维时间序列（RON 损失）
])

model.compile(loss='mean_squared_error', optimizer=tf.optimizers.Adam(0.001), metrics=[tf.metrics.MeanSquaredError()])

print(model.summary())  # 模型结构

history = model.fit(train_X, train_y, epochs=100, batch_size=32, validation_data=(test_X, test_y))

plot_train_loss(history.history['loss'], '')
plot_train_loss(history.history['val_loss'], 'val')

print(model.evaluate(test_X, test_y))

y_pred = model.predict(test_X)
plot_predicted_result(test_y, y_pred, time_step[len(train_X):])

# data = pd.read_excel(r'C:\Users\darkchii\Desktop\1班-1825101001-张三\附件1 区域高程数据(1).xlsx', header=None)
# unit_d = 38.2
# x = unit_d * (np.array(range(data.shape[0])))
# y = unit_d * (np.array(range(data.shape[1])))
# y, x = np.meshgrid(y, x)
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(x, y, data,)
# plt.show()
