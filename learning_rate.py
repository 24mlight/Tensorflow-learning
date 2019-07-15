import tensorflow as tf


LEARNING_RATE_BASE = 0.1 #定义初学习率0.1
LEARNINT_RATE_DECAY = 0.99 #学习率衰减率0.99
LEARNING_RATE_STEP = 1 #每隔1个BATCH_STEP就使学习率下降一次
global_steps = tf.Variable(0,trainable=False) #不可训练，作为LEARNING_RATE_STEP的计数器
w = tf.Variable(5,dtype=tf.float32) #参数w
#正向传播
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,LEARNING_RATE_STEP,LEARNINT_RATE_DECAY)
loss = tf.square(w+1)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_steps)
#逆向传播训练w
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        sess.run(train_step)
        learning_rate_val = sess.run(learning_rate)
        global_step_val = sess.run(global_steps)
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        print("After %s steps:global_step is %s,w=%f,loss=%f" % 
             (i,global_step_val,w_val,loss_val)）

#输出结果
After 0 steps:global_step is 1,w=3.800000,loss=23.040001
After 1 steps:global_step is 2,w=2.840000,loss=14.745600
After 2 steps:global_step is 3,w=2.072000,loss=9.437184
After 3 steps:global_step is 4,w=1.457600,loss=6.039798
After 4 steps:global_step is 5,w=0.966080,loss=3.865470
After 5 steps:global_step is 6,w=0.572864,loss=2.473901
After 6 steps:global_step is 7,w=0.258291,loss=1.583297
After 7 steps:global_step is 8,w=0.006633,loss=1.013310
After 8 steps:global_step is 9,w=-0.194694,loss=0.648518
After 9 steps:global_step is 10,w=-0.355755,loss=0.415052
After 10 steps:global_step is 11,w=-0.484604,loss=0.265633
After 11 steps:global_step is 12,w=-0.587683,loss=0.170005
After 12 steps:global_step is 13,w=-0.670147,loss=0.108803
After 13 steps:global_step is 14,w=-0.736117,loss=0.069634
After 14 steps:global_step is 15,w=-0.788894,loss=0.044566
After 15 steps:global_step is 16,w=-0.831115,loss=0.028522
After 16 steps:global_step is 17,w=-0.864892,loss=0.018254
After 17 steps:global_step is 18,w=-0.891914,loss=0.011683
After 18 steps:global_step is 19,w=-0.913531,loss=0.007477
After 19 steps:global_step is 20,w=-0.930825,loss=0.004785
After 20 steps:global_step is 21,w=-0.944660,loss=0.003063
After 21 steps:global_step is 22,w=-0.955728,loss=0.001960
After 22 steps:global_step is 23,w=-0.964582,loss=0.001254
After 23 steps:global_step is 24,w=-0.971666,loss=0.000803
After 24 steps:global_step is 25,w=-0.977333,loss=0.000514
After 25 steps:global_step is 26,w=-0.981866,loss=0.000329
After 26 steps:global_step is 27,w=-0.985493,loss=0.000210
After 27 steps:global_step is 28,w=-0.988394,loss=0.000135
After 28 steps:global_step is 29,w=-0.990716,loss=0.000086
After 29 steps:global_step is 30,w=-0.992572,loss=0.000055
After 30 steps:global_step is 31,w=-0.994058,loss=0.000035
After 31 steps:global_step is 32,w=-0.995246,loss=0.000023
After 32 steps:global_step is 33,w=-0.996197,loss=0.000014
After 33 steps:global_step is 34,w=-0.996958,loss=0.000009
After 34 steps:global_step is 35,w=-0.997566,loss=0.000006
After 35 steps:global_step is 36,w=-0.998053,loss=0.000004
After 36 steps:global_step is 37,w=-0.998442,loss=0.000002
After 37 steps:global_step is 38,w=-0.998754,loss=0.000002
After 38 steps:global_step is 39,w=-0.999003,loss=0.000001
After 39 steps:global_step is 40,w=-0.999202,loss=0.000001
