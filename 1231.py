import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
a = tf.constant([1,2,3,4,5,6,7,8,9,10,11,12], shape=[3,2,2])
# e = tf.Variable(tf.float32, a)
b = tf.constant([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30], shape=[3,2,5])
c = tf.tensordot(a, b, axes=(2,1))
d = tf.concat([tf.reshape(c[0][0][0], [1,-1]), tf.reshape(c[0][1][0], [1,-1])], 0)
# alphas = tf.nn.softmax(e, name='alphas')  # (B,T) shape
with tf.Session():
    print('a:')
    print(a.eval())
    print('b:')
    print(b.eval())
    print('c:')
    print(c.eval())
    # print('alpha')
    # print(alphas.eval())
    print(c.shape)
    # print(c.eval()[0][0])
    print(d.eval())
# df = pd.read_csv("boston housing price.csv", header=0)
# df = np.array(df.values)  # 获取df的值并且转换成 np 的数组格式
#
# y_data = df[:, 12]  # 标签数据
# for i in range(12):
#     df[:, i] = (df[:, i] - df[:, i].min()) / (df[:, i].max() - df[:, i].min())
# x_data = df[:, :12]  # 特征数据
# x = tf.placeholder(tf.float32, [None, 12], name="x")
# y = tf.placeholder(tf.float32, [None, 1], name="y")
w_omega = tf.Variable(tf.random.normal([12,1], stddev=0.1))
b_omega = tf.Variable(tf.random.normal([1], stddev=0.1))
u_omega = tf.Variable(tf.random.normal([1], stddev=0.1))
# with tf.name_scope('v'):
    # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
    #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
    # v = tf.tensordot(x, w_omega, axes=1) + b_omega           # (16, ?, 50)
    # v = w_omega * inputs
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
# loss_f = tf.reduce_mean(tf.square(y - v))
# train = optimizer.minimize(loss_f)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(50):
        # xs = [i]
        # xs = np.array(xs).reshape(1,1)
        # ys = [i*3]
        # ys = np.array(ys).reshape(1, 1)
        # feed = {x: x_data,
        #         y: y_data}
        # fetches = [v, w_omega]
        # loss_sum = 0.0
        # for xs, ys in zip(x_data, y_data):
        #     xs = xs.reshape(1, 12)
        #     ys = ys.reshape(1, 1)
            # fetches = [v, w_omega]
            # feed={x: xs, y: ys}
            # _, loss = sess.run([train, loss_f], feed_dict={x: xs, y: ys})
        # myv, myw_omega = sess.run(fetches=fetches, feed_dict=feed)
        # if i % 10 == 0:
        #     print(sess.run(w_omega))
            # print("lost: %f" % sess.run(loss, feed_dict=feed))
# x = tf.placeholder(tf.float32)
# W = tf.Variable(tf.zeros([1]))
# b = tf.Variable(tf.zeros([1]))
# y_ = tf.placeholder(tf.float32)
#
# y = W * x + b
#
# lost = tf.reduce_mean(tf.square(y_-y))
# optimizer = tf.train.GradientDescentOptimizer(0.0000001)
# train_step = optimizer.minimize(lost)
#
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
#
# steps = 1000
# for i in range(steps):
#     xs = [i]
#     ys = [3 * i]
#     feed = { x: xs, y_: ys }
#     sess.run(train_step, feed_dict=feed)
#     if i % 100 == 0 :
#         print("After %d iteration:" % i)
#         print("W: %f" % sess.run(W))
#         print("b: %f" % sess.run(b))
#         print("lost: %f" % sess.run(lost, feed_dict=feed))

import matplotlib.pyplot as plt


# 读取数据文件,并且第一行为数据的开头
# df = pd.read_csv("boston housing price.csv", header=0)
# df = np.array(df.values)  # 获取df的值并且转换成 np 的数组格式
#
# y_data = df[:, 12]  # 标签数据
# for i in range(12):
#     df[:, i] = (df[:, i] - df[:, i].min()) / (df[:, i].max() - df[:, i].min())
# x_data = df[:, :12]  # 特征数据
# x = tf.placeholder(tf.float32, [None, 12], name="x")
# y = tf.placeholder(tf.float32, [None, 1], name="y")
# with tf.name_scope("Model1"):
#     # 初始化w为shape=（12,1），服从标准差为0.01的随机正态分布的数
#     w = tf.Variable(tf.random_normal([12, 1], stddev=0.01), name="W")
#     # 初始化b为1.0
#     b = tf.Variable(1.0, name="b")
#     # pred = tf.matmul(x,w)+b
#     pred = tf.tensordot(x, w, axes=1) + b
# with tf.name_scope("LossFunction"):
#     loss_function = tf.reduce_mean(tf.square(y - pred))
# train_epochs = 50  # 迭代次数
# learning_rate = 0.01  # 学习率
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)
# sess = tf.Session()  # 建立会话
# init = tf.global_variables_initializer()  # 变量初始化
# sess.run(init)
# loss_list = []  # 用于保存loss的值
# for epoch in range(train_epochs):
#     loss_sum = 0.0
#     for xs, ys in zip(x_data, y_data):
#         xs = xs.reshape(1, 12)
#         ys = ys.reshape(1, 1)
#         _, loss = sess.run([optimizer, loss_function], feed_dict={x: xs, y: ys})
#         loss_sum = loss_sum + loss  # 累加损失
#
#     x_data, y_data = shuffle(x_data, y_data)  # 打乱数据顺序 避免过拟合假性学习
#
#     b0temp = b.eval(session=sess)
#     w0temp = w.eval(session=sess)
#     loss_average = loss_sum / len(y_data)  # 所有数据的平均损失
#     loss_list.append(loss_average)  # 记录平均损失
#     if epoch % 10 == 0:
#         print(sess.run(w))
# plt.plot(loss_list)  # 显示迭代过程中的平均代价
# plt.show()  # 显示图表

