import tensorflow as tf


state = tf.Variable(0, name='counter')  # define
# print(state.name)
one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)  # define  

init = tf.global_variables_initializer()  # must have if define variable
with tf.Session() as sess:
    sess.run(init)   # initialize
    for _ in range(3): 
        sess.run(update)  # initialize 
        print(sess.run(state))
