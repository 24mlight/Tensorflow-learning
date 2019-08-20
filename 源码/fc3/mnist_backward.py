import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import os

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 40000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "E:\MODEL_SAVE" #修改成你的模型存放地址
MODEL_NAME = "mnist_model"

def backward(mnist):
    x = tf.placeholder(tf.float32,[None,mnist_forward.INPUT_NODE])
    y_= tf.placeholder(tf.float32,[None,mnist_forward.OUTPUT_NODE])
    y = mnist_forward.forward(x,REGULARIZER)
    global_step = tf.Variable(0,trainable = False)
    #tf.argmax(y_,1) 按行计算最大值,并输出最大的索引
    cem = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1)))
    loss = cem + tf.add_n(tf.get_collection('losses'))

    learning_rete = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase = True)

    train_step = tf.train.GradientDescentOptimizer(learning_rete).minimize(loss,global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step,ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        inti_op = tf.global_variables_initializer()
        sess.run(inti_op)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPS):
            xs , ys = mnist.train.next_batch(BATCH_SIZE)
            _ , loss_value , step = sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            if i%1000 == 0 :
                print("After %d train step(s), loss is %f"%(step,loss_value))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

def main():
    mnist = input_data.read_data_sets("./data/",one_hot=True)
    backward(mnist)

if __name__ == '__main__':
    main()




























