import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def nnregress():
    '''生成测试点'''
    x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
    noise = np.random.normal(0, 0.02, x_data.shape)
    y_data = np.square(x_data) + noise

    '''定义两个placeholder'''
    x = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(tf.float32, [None, 1])

    '''定义神经网络中间层  10个神经元 '''
    Weights_L1 = tf.Variable(tf.random_normal([1, 10]))
    Biases_L1 = tf.Variable(tf.zeros([1, 10]))
    Wx_Plus_Bia_L1 = tf.matmul(x, Weights_L1) + Biases_L1
    L1 = tf.nn.tanh(Wx_Plus_Bia_L1)

    '''定义神经网络输出层'''
    Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
    Biases_L2 = tf.Variable(tf.zeros([1, 1]))
    Wx_Plus_Bia_L2 = tf.matmul(L1, Weights_L2) + Biases_L2
    L2 = tf.nn.tanh(Wx_Plus_Bia_L2)

    '''二次代价函数'''
    loss = tf.reduce_mean(tf.square(y - L2))

    '''梯度下降'''
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    '''初始化'''
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for _ in range(2000):
            sess.run(train_step, feed_dict={x: x_data, y: y_data})

        '''预测值'''
        prediction_value = sess.run(L2, feed_dict={x: x_data})
        plt.figure()
        plt.scatter(x_data, y_data)
        plt.plot(x_data, prediction_value, 'r-', lw=5)
        plt.show()


if __name__ == '__main__':
     nnregress()