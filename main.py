# -*- coding: utf-8 -*-

import numpy.linalg as LA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# データセットの読み込み
df = pd.read_csv('test.csv')

print(df)

# dietの値により，一軍(x1)と零軍(x2)に分割
x = df[['height', 'weight', 'dis']].values
x1 = df[df.diet == 1].loc[:, ['height', 'weight', 'dis']]  # 一軍
x2 = df[df.diet == 0].loc[:, ['height', 'weight', 'dis']]  # 零軍

# 判別関数の計算
n1 = len(x1)
n2 = len(x2)
m1 = np.mean(x1, axis=0)
m2 = np.mean(x2, axis=0)
m = (m1*n1+m2*n1)/(n1+n2)

# 判別関数の計算
sw = ((x1-m1).T @ (x1-m1)) + ((x2-m2).T @ (x2-m2))
sinv = LA.inv(sw)
w = -sinv @ (m2 - m1)
print('重み：w = ', w)
print('平均：m = ', m)
Z = w[0]*(df.height-m[0]) + w[1]*(df.weight-m[1]) + w[2]*(df.dis-m[2])
# Z>0の時，ダイエット実施
print(Z)
