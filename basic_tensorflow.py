import tensorflow as tf
import numpy as np


def main():
    # create data
    x_data = np.random.rand(100).astype(np.float32)
    y_data = x_data*0.1 + 0.3

    ### create tensorflow structure start ###
    Weights  = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    biases = tf.Variable(tf.zeros([1]))

    y = Weights*x_data + biases

    loss = tf.reduce_mean(tf.square(y-y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    # 在TensorFlow的世界里，变量的定义和初始化是分开的，
    # 所有关于图变量的赋值和计算都要通过tf.Session的run来进行。
    # 想要将所有图变量进行集体初始化时应该使用tf.global_variables_initializer
    ### create tensorflow structure end ###

    sess = tf.Session()
    sess.run(init)   # very important
    for step in range(201):
        sess.run(train)
        if step % 10 == 0:
            print(step, sess.run(Weights), sess.run(biases))


if __name__ == '__main__':
    main()
