import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
'''数据集'''
mnist = input_data.read_data_sets("MNIST_DATA", one_hot=True)

# 每个批次的大小
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size
'''命名空间'''

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784], name='x_input')
y = tf.placeholder(tf.float32, [None, 10], name='y_input')
# 创建一个简单的神经网络
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

'''代价函数'''
# loss = tf.reduce_mean(tf.square(y-prediction))
'''交叉熵代价函数'''
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
'''梯度下降法'''
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
'''初始化变量'''
init = tf.global_variables_initializer()
'''结果存放在列表中'''
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
'''准确率'''
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as sess:
    sess.run(init)
    # writer = tf.summary.FileWriter('logs/', sess.graph)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter:{}:Testing Accuracy:{}".format(str(epoch), str(acc)))