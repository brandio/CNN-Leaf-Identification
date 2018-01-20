# Author : Branden Strochinsky

# Contains the cnn model and trains/test it on the LeafSnap Dataset
# Usage : Before using this script format_leaf_data in data.py should be ran

import tensorflow as tf
import matplotlib.pyplot as plot
import data
import numpy as np

dataset = data.load_leaf_data()
train_data = dataset[0]
test_data = dataset[1]

n_classes = 10
batch_size = 128

x = tf.placeholder('float', [None, 7500])
y = tf.placeholder('float')


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_nerual_net(x):
    weights = {'w_conv1': tf.Variable(tf.random_normal([5, 5, 3, 24])),
               'w_conv2': tf.Variable(tf.random_normal([5, 5, 24, 48])),
               'w_fc': tf.Variable(tf.random_normal([13*13*48, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = { 'b_conv1': tf.Variable(tf.random_normal([24])),
               'b_conv2': tf.Variable(tf.random_normal([48])),
               'b_fc': tf.Variable(tf.random_normal([1024])),
               'out': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 50, 50, 3])

    conv1 = tf.nn.relu(conv2d(x, weights['w_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['w_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 13*13*48])
    fc = tf.nn.relu(tf.matmul(fc, weights['w_fc']) + biases['b_fc'])

    output = tf.matmul(fc, weights['out']) + biases['out']
    return output

def train_neural_network(x):
    prediction = convolutional_nerual_net(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=.01).minimize(cost)

    hm_epochs = 50

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        loss = []
        for epoch in range(hm_epochs):
            epoch_loss = 0

            for index in range(int(train_data['images'].shape[0]/ batch_size)):
                epoch_x, epoch_y = next_batch(batch_size,train_data['images'],train_data['labels'])
                index, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            loss.append(epoch_loss)
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        plot.plot(range(len(loss)),loss)
        plot.ylabel("Error")
        plot.xlabel("Epoch")
        plot.show()

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: test_data['images'], y: test_data['labels']}))


train_neural_network(x)

