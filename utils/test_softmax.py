import tensorflow as tf
import numpy as np

inti_a_na = np.random.rand(2,2,5)
a = tf.convert_to_tensor(inti_a_na, dtype=tf.float32)
a_trans = tf.transpose(a, [0,2,1])
b = tf.nn.softmax(a_trans, axis=1)
b = tf.transpose(b, [0,2,1])
c = tf.nn.softmax(a, axis=2)
with tf.Session() as sess:
    print(inti_a_na)
    print(b.eval())
    print(c.eval())
    # inti_b_na = np.random.rand(10,5)
    # print(inti_b_na)
    # res = sess.run(c, {b:inti_b_na})
    # print(res)