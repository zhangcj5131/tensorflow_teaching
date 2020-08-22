import tensorflow as tf

def test_softmax():
    with tf.Graph().as_default():
        logits = tf.placeholder(tf.float32)
        softmax = tf.nn.softmax(logits)
        prediction = tf.math.argmax(softmax)
        with tf.Session() as sess:
            input_logits = [1,2,3]
            print('softmax={}'.format(sess.run(softmax, {logits: input_logits})))
            print('prediction={}'.format(sess.run(prediction, {logits: input_logits})))

def test_cross_entropy():
    with tf.Graph().as_default():
        logits = tf.placeholder(tf.float32)
        one_hot = tf.placeholder(tf.float32)
        label = tf.placeholder(tf.int32)

        softmax = tf.nn.softmax(logits)

        cross_entropy1 = -tf.reduce_sum(tf.log(softmax) * one_hot)
        cross_entropy2 = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                 labels=one_hot)
        cross_entropy3 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                        labels=label)
        with tf.Session() as sess:
            input_logit=[1,2,3]
            input_one_hot = [0,0,1]
            input_label = 2
            print('cross_entropy1={}'.format(sess.run(cross_entropy1, {logits: input_logit,
                                                              one_hot: input_one_hot})))
            print('cross_entropy2={}'.format(sess.run(cross_entropy2, {logits: input_logit,
                                                              one_hot: input_one_hot})))
            print('cross_entropy3={}'.format(sess.run(cross_entropy3, {logits: input_logit,
                                                              label: input_label})))






if __name__ == '__main__':
    test_cross_entropy()