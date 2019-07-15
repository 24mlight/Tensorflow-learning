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
             (i,global_step_val,w_val,loss_val))
