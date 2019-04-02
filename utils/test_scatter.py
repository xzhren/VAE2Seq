import tensorflow as tf
import numpy as np

batch_size = 5
vocab_size = 100
time_step = 10
decoder_time = 8
updates = np.random.rand(batch_size, time_step)
# print(updates)
print(updates.shape)

shape = [batch_size, vocab_size]
print(shape)

encoder = np.random.randint(vocab_size, size=(batch_size, time_step))
# print(encoder)
print(encoder.shape)


batch_nums = tf.range(0, limit=batch_size)
batch_nums = tf.expand_dims(batch_nums, 1)
batch_nums = tf.tile(batch_nums, [1, time_step])
indices = tf.stack( (batch_nums, encoder), axis=2) # shape (batch_size, enc_t, 2)
scatter = tf.scatter_nd(indices, updates, shape)
with tf.Session() as sess:
    res = sess.run(scatter)
    print(res)
    print(res.shape)
    print (indices.shape)
    print (updates.shape)