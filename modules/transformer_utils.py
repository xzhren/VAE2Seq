import tensorflow as tf

def trans_dist_embedding(latent_size, encoder_class_num, decoder_class_num, input_mean, input_logvar):
    # input: b x l
    # [o]: predition: b x l
    with tf.variable_scope('trans_encoder'):
        trans_encoder_embedding = tf.get_variable('trans_encoder_embedding',
            [encoder_class_num, latent_size],
            tf.float32)
        trans_encoder_embedding_var = tf.get_variable('trans_encoder_embedding_var',
            [encoder_class_num, latent_size],
            tf.float32)
        input_dist_mean = tf.nn.softmax(tf.matmul(input_mean, trans_encoder_embedding, transpose_b=True)) # b x encoder_class_num
        input_dist_logvar = tf.nn.softmax(tf.matmul(input_logvar, trans_encoder_embedding_var, transpose_b=True)) # b x encoder_class_num
    with tf.variable_scope('trans'):
        trans_w = tf.get_variable('trans_w', [encoder_class_num, decoder_class_num], tf.float32)
        trans_w_var = tf.get_variable('trans_w_var', [encoder_class_num, decoder_class_num], tf.float32)
        input_dist_mean = tf.nn.softmax(tf.matmul(input_dist_mean, trans_w)) # b x decoder_class_num
        input_dist_logvar = tf.nn.softmax(tf.matmul(input_dist_logvar, trans_w_var)) # b x decoder_class_num
    with tf.variable_scope('trans_decoder'):
        trans_decoder_embedding = tf.get_variable('trans_decoder_embedding',
            [decoder_class_num, latent_size],
            tf.float32)
        trans_decoder_embedding_var = tf.get_variable('trans_decoder_embedding_var',
            [decoder_class_num, latent_size],
            tf.float32)
        predition_mean = tf.matmul(input_dist_mean, trans_decoder_embedding) # b x l
        predition_logvar = tf.matmul(input_dist_logvar, trans_decoder_embedding_var) # b x l
        predition = predition_mean + tf.exp(0.5 * predition_logvar) * tf.truncated_normal(tf.shape(predition_logvar))

    return predition_mean, predition_logvar, predition

def trans_vector_embedding(latent_size, encoder_class_num, decoder_class_num, input_z):
    # input: b x l
    # [o]: predition: b x l
    with tf.variable_scope('trans_encoder'):
        trans_encoder_embedding = tf.get_variable('trans_encoder_embedding',
            [encoder_class_num, latent_size],
            tf.float32)
        input_dist_z = tf.nn.softmax(tf.matmul(input_z, trans_encoder_embedding, transpose_b=True)) # b x encoder_class_num
    with tf.variable_scope('trans'):
        trans_w = tf.get_variable('trans_w', [encoder_class_num, decoder_class_num], tf.float32)
        input_dist_z = tf.nn.softmax(tf.matmul(input_dist_z, trans_w)) # b x decoder_class_num
    with tf.variable_scope('trans_decoder'):
        trans_decoder_embedding = tf.get_variable('trans_decoder_embedding',
            [decoder_class_num, latent_size],
            tf.float32)
        predition_z = tf.matmul(input_dist_z, trans_decoder_embedding) # b x l
    return predition_z

def trans_dist_mlp(latent_size, input_mean, input_logvar):
    # input: b x l
    # [o]: predition: b x l
    with tf.variable_scope('trans_mlp'):
        trans_mean = tf.layers.dense(input_mean, latent_size, tf.nn.tanh, name="tran_mean_1")
        trans_mean = tf.layers.dense(trans_mean, latent_size, tf.nn.tanh, name="tran_mean_2")
        trans_mean = tf.layers.dense(trans_mean, latent_size, tf.nn.tanh, name="tran_mean_3")

        trans_logvar = tf.layers.dense(input_logvar, latent_size, tf.nn.tanh, name="trans_logvar_1")
        trans_logvar = tf.layers.dense(trans_logvar, latent_size, tf.nn.tanh, name="tran_logvar_2")
        trans_logvar = tf.layers.dense(trans_logvar, latent_size, tf.nn.tanh, name="tran_logvar_3")

        predition_mean = tf.layers.dense(trans_mean, latent_size, name="tran_mean") # b x l
        predition_logvar = tf.layers.dense(trans_logvar, latent_size, name="tran_logvar") # b x l
        predition = predition_mean + tf.exp(0.5 * predition_logvar) * tf.truncated_normal(tf.shape(predition_logvar))
    return predition_mean, predition_logvar, predition

def trans_vector_mlp(latent_size, input_z):
    # input: b x l
    # [o]: predition: b x l
    with tf.variable_scope('trans_mlp'):
        trans_z = tf.layers.dense(input_z, latent_size, tf.nn.tanh, name="tran_z_1")
        trans_z = tf.layers.dense(trans_z, latent_size, tf.nn.tanh, name="tran_z_2")
        trans_z = tf.layers.dense(trans_z, latent_size, tf.nn.tanh, name="tran_z_3")

        predition = tf.layers.dense(trans_z, latent_size, name="tran_z") # b x l
    return predition