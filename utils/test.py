import tensorflow as tf
with tf.variable_scope("scope1") as scope1:
    with tf.variable_scope("/scope2") as scope2:
        print(scope2.name)

import numpy as np

inti_a_na = np.random.rand(10,5)
print(inti_a_na, inti_a_na.shape)
a = tf.convert_to_tensor(inti_a_na, dtype=tf.float32)
b = a + 1
c = b + 1
with tf.Session() as sess:
    print(c.eval())
    inti_b_na = np.random.rand(10,5)
    print(inti_b_na)
    res = sess.run(c, {b:inti_b_na})
    print(res)