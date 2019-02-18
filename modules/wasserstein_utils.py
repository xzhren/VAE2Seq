import tensorflow as tf

def wasserstein_loss(source_dist, target_dist):
    """
        source_dist: b x c
        target_dist: b x c
    """
    batch_size = tf.shape(source_dist)[0]
    feature_num = tf.shape(source_dist)[1]
    
    source_dist = tf.tile(tf.expand_dims(source_dist, [2]), [1,1,feature_num]) # b x c x c
    target_dist = tf.tile(tf.expand_dims(target_dist, [2]), [1,1,feature_num]) # b x c x c
    target_dist = tf.transpose(target_dist, [0,2,1])

    feature_num_tensor = tf.cast(tf.range(feature_num), tf.float32) / tf.cast(feature_num, tf.float32)# c
    feature_num_tensor = tf.tile(tf.expand_dims(feature_num_tensor, [0]), [batch_size,1]) # b x c
    source_index_dist = tf.tile(tf.expand_dims(feature_num_tensor, [2]), [1,1,feature_num]) # b x c x c
    target_index_dist = tf.tile(tf.expand_dims(feature_num_tensor, [2]), [1,1,feature_num]) # b x c x c
    target_index_dist = tf.transpose(target_index_dist, [0,2,1])
    
    wasserstein_dist = tf.reduce_sum( tf.abs( (source_index_dist - target_index_dist) * (source_dist - target_dist) ), [1,2])
    wasserstein_dist = wasserstein_dist / tf.reduce_sum( tf.abs(source_dist - target_dist), [1,2]) # b
    
    wasserstein_dist = tf.reduce_mean(wasserstein_dist) # 1
    return wasserstein_dist
    