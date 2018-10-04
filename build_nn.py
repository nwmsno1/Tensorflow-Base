import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, insize, outsize, activation_function=None):
    Weights = tf.Variable(tf.random_normal([insize, outsize]))
    biases = tf.Variable(tf.zeros([1, outsize]))
    Wx_puls_b = tf.matmul(inputs, Weights) + biases

    if activation_function is None:
        outputs = Wx_puls_b
    else:
        outputs = activation_function(Wx_puls_b)

    return outputs


def build_nn():
    x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
    y_data = np.square(x_data) - 0.5 + noise
    '''利用占位符定义我们所需的神经网络的输入。 tf.placeholder()就是代表占位符，
    这里的None代表无论输入有多少都可以，因为输入只有一个特征，所以这里是1'''
    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])

    # define hidden layer
    l1 = add_layer(x_data, 1, 10, activation_function=tf.nn.relu)
    # define output layer
    prediction = add_layer(l1, 10, 1, activation_function=None)
    # compute the deviation between prediction and y_data
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(prediction-ys), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    init = tf.global_variables_initializer()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # s表示点点的大小，c就是color嘛，marker就是点点的形状哦o,x,*><^,都可以
    # alpha,点点的亮度，label，标签
    ax.scatter(x_data, y_data, marker='x')
    # plt.ion()用于连续显示
    plt.ion()
    plt.show()
    with tf.Session() as sess:
        sess.run(init)
        # train
        for i in range(1000):
            sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
            # see the step improvement
            if i % 50 == 0:
                print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
                try:
                    ax.lines.remove(lines[0])
                except Exception:
                    pass
                prediction_value = sess.run(prediction, feed_dict={xs: x_data})
                # plot the prediction
                lines = ax.plot(x_data, prediction_value, 'r-', lw=2)
                plt.pause(0.5)


def main():
    build_nn()


if __name__ == '__main__':
    main()
