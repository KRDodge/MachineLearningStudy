import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt

x_train = [1,2,3]
y_train = [1,2,3]

W = tf.Variable(5.0)
b = tf.Variable(tf.random_normal([1]), name='bias')
X = tf.constant([1.0, 2.0, 3.0])
Y = tf.constant([1.0, 2.0, 3.0])

hypothesis = X * W
cost = tf.reduce_mean(tf.square(hypothesis - Y))

learning_rate = 0.01
gradient = tf.reduce_mean((W*X - Y) * X) * 2
descent = W - learning_rate * gradient
update = W.assign(descent)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

gvs = optimizer.compute_gradients(cost)
apply_gradients = optimizer.apply_gradients(gvs)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run([gradient, W]), gvs)
    sess.run(apply_gradients)

