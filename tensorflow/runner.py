from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# load mnist data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

num_iteration = 10000

# set placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# set parameters
W = tf.get_variable("W", [784, 10], initializer=tf.contrib.layers.xavier_initializer(seed=1))
b = tf.Variable(tf.zeros([10]))

# forward propagation
Z_L = tf.nn.softmax(tf.add(tf.matmul(x, W), b))

# compute cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=Z_L))

# define optimizer
optimizer = tf.train.AdamOptimizer().minimize(cost)

# initialize parameters
init = tf.global_variables_initializer()


# start session to compute tensorflow graph
with tf.Session() as sess:
    sess.run(init)

    for iteration in range(num_iteration):
        mini_X, mini_Y = mnist.train.next_batch(100)
        _, minibatch_cost = sess.run([optimizer, cost], feed_dict={x:mini_X, y:mini_Y})

        if iteration % 1000 == 0:
            print("iteration: {}, minibatch_cost: {}".format(iteration, minibatch_cost))

    correct_prediction = tf.equal(tf.argmax(Z_L, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    test_accuracy = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
    print("accuracy for test data: ", test_accuracy)