import os
import numpy as np
import tensorflow as tf
import time
import pandas as pd
import cv2

data_csv = pd.read_csv("train_new.csv")
train_x = data_csv['file_name']
train_y = data_csv['labels']

F = 16
L = 3
S = 2
D = 256
INPUT_WIDTH = 48
INPUT_HEIGHT = 27
CHECKPOINT_PATH = None


def shape_text(tensor):
    return ", ".join(["?" if i is None else str(i) for i in tensor.get_shape().as_list()])


def conv3d(inp, filters, dilation_rate):
    return tf.keras.layers.Conv3D(filters, kernel_size=3, dilation_rate=(dilation_rate, 1, 1),
                                  padding="SAME", activation=tf.nn.relu, use_bias=True,
                                  name="Conv3D_{:d}".format(dilation_rate))(inp)


inputs = tf.placeholder(tf.uint8,
                        shape=[None, None, INPUT_HEIGHT, INPUT_WIDTH, 3])

net = tf.cast(inputs, dtype=tf.float32) / 255.

labels = tf.placeholder(tf.float32, shape=[None, 2])
y_true_cls = tf.argmax(labels, dimension=1)

for idx_l in range(L):
    filters = (2 ** idx_l) * F

    for idx_s in range(S):
        net = tf.identity(net)  # improves look of the graph in TensorBoard
        conv1 = conv3d(net, filters, 1)
        conv2 = conv3d(net, filters, 2)
        conv3 = conv3d(net, filters, 4)
        conv4 = conv3d(net, filters, 8)
        net = tf.concat([conv1, conv2, conv3, conv4], axis=4)

    net = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2))(net)

shape = [tf.shape(net)[0], tf.shape(net)[1],
         np.prod(net.get_shape().as_list()[2:])]
net = tf.reshape(net, shape=shape, name="flatten_3d")
net = tf.keras.layers.Dense(D, activation=tf.nn.relu)(net)

logits = tf.keras.layers.Dense(2, activation=None)(net)
predictions = tf.nn.softmax(logits, name="predictions")[:, :, 1]

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=labels)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

correct_prediction = tf.equal(predictions, labels)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

num_epochs = 30
batch_size = 20


with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Loop over number of epochs
    for epoch in range(num_epochs):

        start_time = time.time()
        train_accuracy = 0

        for batch in range(0, int(len(data_csv['labels'])/batch_size)):

            # Get a batch of images and labels
            x_batch_images = train_x[batch*batch_size: (batch+1)*batch_size]
            x_batch = []

            for x in x_batch_images:
                img = cv2.imread(os.path.join("../test_frames", x))
                img = cv2.resize(img,(INPUT_HEIGHT, INPUT_WIDTH))
                img = np.reshape(img, [-1, INPUT_HEIGHT, INPUT_WIDTH, 3])
                x_batch.append(img)

            y_true_batch = train_y[batch*batch_size: (batch+1)*batch_size]
            # Put the batch into a dict with the proper names for placeholder variables
            feed_dict_train = {net: x_batch, labels: y_true_batch}

            # Run the optimizer using this batch of training data.
            sess.run(optimizer, feed_dict=feed_dict_train)

            # Calculate the accuracy on the batch of training data
            train_accuracy += sess.run(accuracy, feed_dict=feed_dict_train)

        train_accuracy /= int(len(data_csv['labels'])/batch_size)

        end_time = time.time()

        print("Epoch "+str(epoch+1)+" completed : Time usage " +
              str(int(end_time-start_time))+" seconds")
        print("Training Accuracy: {}".format(train_accuracy))
