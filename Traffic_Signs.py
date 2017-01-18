# import

# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
# import cv2
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelBinarizer
import matplotlib.image as mpimg

# config
num_iterations = 1000

# processing

# def grayscale(img):
#     """Applies the Grayscale transform
#     This will return an image with only one color channel
#     but NOTE: to see the returned image as grayscale
#     you should call plt.imshow(gray, cmap='gray')"""
#     return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def normalize(X):
    a,b,xmin,xmax = 0.1,0.9,0,255
    return a+(X-xmin)*(b-a)/(xmax-xmin)

# def grayscale_4d(X):
#     X_gray = np.empty(X[:,:,:,0].shape)
#     for i in range(len(X_gray)):
#         X_gray[i] = normalize_grayscale(grayscale(X[i]))
#     return(X_gray)

def flat_3d(X):
    n, a,b = X.shape[0],X.shape[1],X.shape[2]
    X_flat = np.empty((n,a*b))
    for i in range(len(X_flat)):
        X_flat[i] = X[i].reshape(a*b)
    return(X_flat)

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def flatten_x(x):
    x_shape = x.shape
    num_features = np.prod(x_shape[1:4])
    x_flat = np.reshape(x, [-1, num_features])
    return x_flat, num_features


# Load pickled data
import pickle

training_file = 'lab 2 data/train.p'
testing_file = 'lab 2 data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

# images and class number
X_train, y_train_cls = train['features'], train['labels']
X_test, y_test_cls = test['features'], test['labels']

# shuffle the training data
X_train,y_train_cls = shuffle_in_unison(X_train, y_train_cls)

X_train = normalize(X_train)

X_train_flat, X_train_flat_features = flatten_x(X_train)

encoder = LabelBinarizer()
encoder.fit(y_train_cls)
y_train = encoder.transform(y_train_cls).astype(np.float32)
y_test = encoder.transform(y_test_cls).astype(np.float32)


# basic data summary.

n_train = len(y_train_cls)
n_test = len(y_test_cls)
num_channels = 3
img_size = 32
img_shape = (img_size, img_size, num_channels)
num_classes = len(Counter(y_train_cls)) #43

# weight/bias helpers

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

# layer helpers

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=False): # Use Rectified Linear Unit (ReLU)?

    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    layer = tf.nn.relu(layer)
    return layer, weights

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

# placeholders

x = tf.placeholder(tf.float32, [None, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])
keep_prob = tf.placeholder("float")

# model

# First convolutional layer
conv1_filter_size = 5
conv1_num_filters = 16

layer_conv1, weights_conv1 = new_conv_layer(x,              # The previous layer.
                   num_channels, # Num. channels in prev. layer.
                   conv1_filter_size,        # Width and height of each filter.
                   conv1_num_filters,        # Number of filters.
                   use_pooling=True)  # Use 2x2 max-pooling.

layer_dropout1 = tf.nn.dropout(layer_conv1, keep_prob)

# Second convolutional layer

conv2_filter_size = 5
conv2_num_filters = 24

layer_conv2, weights_conv2 = new_conv_layer(layer_dropout1,              # The previous layer.
                   conv1_num_filters, # Num. channels in prev. layer.
                   conv2_filter_size,        # Width and height of each filter.
                   conv2_num_filters,        # Number of filters.
                   use_pooling=True)  # Use 2x2 max-pooling.

layer_dropout2 = tf.nn.dropout(layer_conv2, keep_prob)

# Third convolutional layer

conv3_filter_size = 5
conv3_num_filters = 32

layer_conv3, weights_conv3 = new_conv_layer(layer_dropout2,              # The previous layer.
                   conv2_num_filters, # Num. channels in prev. layer.
                   conv3_filter_size,        # Width and height of each filter.
                   conv3_num_filters,        # Number of filters.
                   use_pooling=True)  # Use 2x2 max-pooling.

layer_dropout3 = tf.nn.dropout(layer_conv3, keep_prob)

# Flatten second conv layer for FC layer

layer_flat, num_features = flatten_layer(layer_dropout3)

# First FC layer

fc1_out_features = 32

layer_fc1 = new_fc_layer(layer_flat,          # The previous layer.
                 num_features,     # Num. inputs from prev. layer.
                 fc1_out_features,    # Num. outputs.
                 use_relu=True) # Use Rectified Linear Unit (ReLU)?

# Last FC layer

layer_last = new_fc_layer(layer_fc1,          # The previous layer.
                 fc1_out_features,     # Num. inputs from prev. layer.
                 num_classes,    # Num. outputs.
                 use_relu=False) # Use Rectified Linear Unit (ReLU)?

# get softmax and predicted class

y_pred = tf.nn.softmax(layer_last)
y_pred_cls = tf.argmax(layer_last, dimension=1) # was y_pred


# cost function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_last, labels=y_true)
cost = tf.reduce_mean(cross_entropy)

# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# accuracy
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# create session and initialize variables
session = tf.Session()
session.run(tf.initialize_all_variables())

# training function
batch_size = 512

def optimize(num_iterations):
    for i in range(num_iterations):
        
        batch_start = i*batch_size % len(X_train)
        batch_end = min(batch_start+batch_size,len(X_train))
        x_batch = X_train[batch_start:batch_end]
        y_true_batch = y_train[batch_start:batch_end]
        y_true_batch_cls = y_train_cls[batch_start:batch_end]
        
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch,
                           keep_prob: 0.5}

        _, batch_cost = session.run([optimizer, cost], feed_dict=feed_dict_train)
        
        if i % 100 == 0:
            feed_dict_train_accuracy = {x: x_batch, y_true: y_true_batch, y_true_cls: y_true_batch_cls, keep_prob: 1.0}
            acc = session.run(accuracy, feed_dict=feed_dict_train_accuracy)
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}, Batch Cost: {2}"
            print(msg.format(i + 1, acc, batch_cost))

# test acuracy

feed_dict_test = {x: X_test,
                  y_true: y_test,
                  y_true_cls: y_test_cls,
                  keep_prob: 1.0}

def print_accuracy():
    acc,correct,pred_cls,pred = session.run([accuracy,correct_prediction, y_pred_cls, y_pred], feed_dict=feed_dict_test)
    print("Accuracy on test-set: {0:.1%}".format(acc))
    print(pred[0])

# training

print('pre-training accuracy: ')
print_accuracy()

print('accuracy:')
optimize(num_iterations=num_iterations)
print_accuracy()

# my images
img1 = mpimg.imread('img1.jpg')
img2 = mpimg.imread('img2.jpg')
img3 = mpimg.imread('img3.jpg')
img4 = mpimg.imread('img4.jpg')
img5 = mpimg.imread('img5.jpg')
X_new = np.concatenate((img1,img2,img3,img4,img5), axis=0).reshape(-1,32,32,3)
y_new_cls = np.array([14,26,14,4,38])
encoder = LabelBinarizer()
encoder.fit(y_train_cls)
y_new = encoder.transform(y_new_cls).astype(np.float32)

feed_dict_new = {x: X_new,
                  y_true: y_new,
                  y_true_cls: y_new_cls,
                  keep_prob: 1.0}

correct, pred_cls, pred = session.run([correct_prediction, y_pred_cls, y_pred], feed_dict=feed_dict_new)

# shows true if predicted class for each image was correct
print(correct)
# shows predicted classes
print(pred_cls)
# shows softmax probabilities, which are very close to 1 and 0. Note that if you have less training iterations, these probabilitse are more diffuse, so I don't know if this is a bug...
print(pred)

# shows top k classes
top_k_preds = session.run(tf.nn.top_k(y_pred, k=3),feed_dict=feed_dict_new)
print(top_k_preds.values)
print(top_k_preds.indices)