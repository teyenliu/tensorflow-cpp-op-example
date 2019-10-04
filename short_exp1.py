import tensorflow as tf 
import numpy as np

session = tf.Session()

n = 2
m = 3
x = tf.placeholder(tf.float32, shape = (n))
W = tf.placeholder(tf.float32, shape = (m, n))

Wx_tf = tf.matmul(W, tf.reshape(x, [-1, 1]))

grad_x_tf = tf.gradients(Wx_tf, x)
grad_W_tf = tf.gradients(Wx_tf, W)

x_rand = np.random.randint(10, size = (n))
W_rand = np.random.randint(10, size = (m, n))

gradient_tf = session.run([grad_x_tf, grad_W_tf], feed_dict = {x: x_rand, W: W_rand})

graph = tf.get_default_graph()
writer = tf.summary.FileWriter("./short_graph_events")
writer.add_graph(graph=graph)

session.close()

print "x_rand: ", x_rand 
print "W_rand: ", W_rand
print "gradient_tf: ", gradient_tf

