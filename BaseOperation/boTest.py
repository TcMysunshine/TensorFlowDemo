import tensorflow as tf
import numpy as np

def constant_test():
    m1 = tf.constant([[1, 3], [2, 4]])
    m2 = tf.constant([[4, 5], [8, 9]])
    product = tf.matmul(m1, m2)
    # print(product)
    with tf.Session() as sess:
        result = sess.run(product)
        print(result)


def variable_test():
    m1 = tf.Variable([1, 3])
    m2 = tf.constant([4, 5])
    sub = tf.subtract(m1, m2)
    add = tf.add(m1, m2)
    init = tf.global_variables_initializer()
    # print(product)
    with tf.Session() as sess:
        sess.run(init)
        sub = sess.run(sub)
        add = sess.run(add)
        print(sub)
        print(add)


'''fetch就是一个sesion里面运行多个OP'''


def fetchAndFeed():
    a = tf.constant(1)
    b = tf.constant(2)
    c = tf.multiply(a, b)
    d = tf.add(a,b)
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    output = tf.multiply(input1,input2)
    with tf.Session() as sess:
        result = sess.run([c, d])
        print(result)
        print(sess.run(output, feed_dict={input1: [7.0], input2:[9.0]}))


def gradientDescend():
    '''生成一百个随机点'''
    x_data = np.random.rand(100)
    y_data = x_data * 0.1 + 0.2

    '''构建线性模型'''
    k = tf.Variable(0.0)
    b = tf.Variable(0.0)
    y = x_data * k + b

    '''代价函数'''
    loss = tf.reduce_mean(tf.square(y-y_data))

    '''梯度下降法作为优化器'''
    optimizer = tf.train.GradientDescentOptimizer(0.1)

    '''最小化代价函数'''
    train = optimizer.minimize(loss)

    '''初始化变量'''
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(200):
            sess.run(train)
            if step % 20 == 0:
                print(step, sess.run([k, b]))


if __name__ == '__main__':
    # constant_test()
    # variable_test()
    # fetchAndFeed()
    # gradientDescend()
    Weights_L1 = tf.Variable(tf.random_normal([1, 10]))
    Biases_L1 = tf.Variable(tf.zeros([1, 10]))
    result = tf.add(Weights_L1, Biases_L1)
    # Weights_L1
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(result))


