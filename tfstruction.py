import tensorflow as tf 
import numpy as np

#准备阶段
BATCH_SISE = 8
SEED = 23455
rdm = np.random.RandomState(SEED)
#ran一个数据集
X = rdm.rand(32,2)
Y = [[int(x0+x1<1)] for (x0,x1) in X]

#正向传播 
#给训练集占个位
x = tf.placeholder(tf.float32, shape=(None,2))
y_ = tf.placeholder(tf.float32, shape=(None,1))
#权重
w1 = tf.Variable(tf.random_normal([2,3],stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1, seed=1))
#输出
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#反向传播
#定义损失函数和最优化函数
loss = tf.reduce_mean(tf.square(y-y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#输出未经过训练的参数w1、w2
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("w1:\n",sess.run(w1))
    print("w2:\n",sess.run(w2))
    print('\n')

#通过迭代训练参数
    STEPS=3000
    for i in range(STEPS):
        start = (i*BATCH_SIZE)%32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_:Y[start:end]})
        if i%500==0:    
            total_loss = sess.run(loss, feed_dict={x:X[start:end], y_:Y[start:end]})
            print("经过",i,"次训练后","损失函数的值为：",total_loss)
    print('\n')
    print('w1:\n',sess.run(w1))
    print('w2:\n',sess.run(w2))


