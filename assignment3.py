#%%

import numpy as np
from numpy import ndarray
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#%% md

#### Задание 1. Реализуйте нейронную сеть с двумя сверточными слоями, и одним полносвязным с нейронами с кусочно-линейной функцией активации. Какова точность построенное модели?

#%%

pickle_file = 'data/notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset: ndarray = save['train_dataset']
    train_labels: ndarray = save['train_labels']
    valid_dataset: ndarray = save['valid_dataset']
    valid_labels: ndarray = save['valid_labels']
    test_dataset: ndarray = save['test_dataset']
    test_labels: ndarray = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

#%%

image_size = 28
num_labels = 10
num_channels = 1  # grayscale
batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64


#%%

def reformat(dataset, labels):
    dataset = dataset.reshape(
        (-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


#%%

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


#%%

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

#%%

graph = tf.Graph()

with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))

    layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    # // equals math.floor
    layer3_weights = tf.Variable(
        tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

    layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))


    # Model.
    def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases


    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

#%%

num_steps = 1001
# num_steps = 2001

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()

    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

        if step % 50 == 0:
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))

    print('Original Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
    first_preds = test_prediction.eval()

#%% md

#### Задание 2. Замените один из сверточных слоев на слой, реализующий операцию пулинга (Pooling) с функцией максимума или среднего. Как это повлияло на точность классификатора?

#%%

graph = tf.Graph()

with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))

    layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

    layer3_weights = tf.Variable(
        tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

    layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))


    # Model.
    def model(data):
        conv1 = tf.nn.relu(tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME') + layer1_biases)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv2 = tf.nn.relu(tf.nn.conv2d(pool1, layer2_weights, [1, 1, 1, 1], padding='SAME') + layer2_biases)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        shape = pool2.get_shape().as_list()
        reshape = tf.reshape(pool2, [shape[0], shape[1] * shape[2] * shape[3]])
        fc1 = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(fc1, layer4_weights) + layer4_biases


    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

#%%

# num_steps = 2001
num_steps = 1001

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()

    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

        if step % 50 == 0:
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))

    print('Max pool Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
    second_preds = test_prediction.eval()

#%% md

#### Задание 3. Реализуйте классическую архитектуру сверточных сетей LeNet-5 (http://yann.lecun.com/exdb/lenet/).

#%%

graph = tf.Graph()

with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    global_step = tf.Variable(0)  # count the number of steps taken.

    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))

    layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

    layer3_weights = tf.Variable(
        tf.truncated_normal([image_size // 4 * image_size // 4 * depth, 120], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[120]))

    layer4_weights = tf.Variable(tf.truncated_normal([120, 84], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[84]))

    layer5_weights = tf.Variable(tf.truncated_normal([84, num_labels], stddev=0.1))
    layer5_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))


    # Model.
    def model(data, keep_prob):
        conv1 = tf.nn.relu(tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME') + layer1_biases)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv2 = tf.nn.relu(tf.nn.conv2d(pool1, layer2_weights, [1, 1, 1, 1], padding='SAME') + layer2_biases)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        shape = pool2.get_shape().as_list()
        reshape = tf.reshape(pool2, [shape[0], shape[1] * shape[2] * shape[3]])

        fc1 = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        fc2 = tf.nn.relu(tf.matmul(fc1, layer4_weights) + layer4_biases)

        y_conv = tf.nn.softmax(tf.matmul(fc2, layer5_weights) + layer5_biases)

        return y_conv


    # Training computation.
    y_conv = model(tf_train_dataset, 0.5)
    y_ = tf_train_labels
    loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

    # Optimizer.
    learning_rate = tf.train.exponential_decay(1e-1, global_step, num_steps, 0.7, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
    train_prediction = y_conv
    valid_prediction = model(tf_valid_dataset, 1.0)
    test_prediction = model(tf_test_dataset, 1.0)

#%%

num_steps = 1001
# num_steps = 5001

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()

    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

        if step % 300 == 0:
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))

    print('Lenet 5 Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
    third_preds = test_prediction.eval()


#%% md

#### Задание 4. Сравните максимальные точности моделей, построенных в лабораторных работах 1-3.  Как можно объяснить полученные различия?

#%%

def plot_diffs(dataset, sample_size):
    fig: plt.Figure = plt.figure()

    sample_images = dataset[np.random.choice(dataset.shape[0], sample_size * sample_size)]
    sample_images = sample_images.reshape(sample_size, sample_size, *dataset.shape[-2:])

    for image_i in range(sample_size):
        for image_j in range(sample_size):
            sample_image = sample_images[image_i][image_j]
            ax = fig.add_subplot(sample_size, sample_size, image_i * sample_size + (image_j + 1))
            ax.imshow(sample_image)
            ax.set_axis_off()

    plt.show()


#%%

print(accuracy(first_preds, test_labels), np.unique(first_preds), first_preds.shape)
print(accuracy(second_preds, test_labels), np.unique(second_preds), second_preds.shape)
print(accuracy(third_preds, test_labels), np.unique(third_preds), third_preds.shape)

first_false_indices = np.argwhere(np.argmax(first_preds, axis=1) != np.argmax(test_labels, axis=1)).flatten()
first_true_indices = np.argwhere(np.argmax(first_preds, axis=1) == np.argmax(test_labels, axis=1)).flatten()
second_false_indices = np.argwhere(np.argmax(second_preds, axis=1) != np.argmax(test_labels, axis=1)).flatten()
second_true_indices = np.argwhere(np.argmax(first_preds, axis=1) == np.argmax(test_labels, axis=1)).flatten()
third_false_indices = np.argwhere(np.argmax(third_preds, axis=1) != np.argmax(test_labels, axis=1)).flatten()
third_true_indices = np.argwhere(np.argmax(first_preds, axis=1) == np.argmax(test_labels, axis=1)).flatten()

first_true_dataset = test_dataset[first_true_indices]
second_true_dataset = test_dataset[second_true_indices]
third_true_dataset = test_dataset[third_true_indices]

first_diff_dataset = test_dataset[np.setdiff1d(first_false_indices, second_false_indices)]
second_diff_dataset = test_dataset[np.setdiff1d(second_false_indices, third_false_indices)]

# Reshape
first_true_dataset = first_true_dataset.reshape(first_true_dataset.shape[:3])
second_true_dataset = second_true_dataset.reshape(second_true_dataset.shape[:3])
third_true_dataset = third_true_dataset.reshape(third_true_dataset.shape[:3])
first_diff_dataset = first_diff_dataset.reshape(first_diff_dataset.shape[:3])
second_diff_dataset = second_diff_dataset.reshape(second_diff_dataset.shape[:3])

#%%

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    a = test_prediction.eval()

#%% md

# Первая сеть с двумя сверточными слоями и одним полносвязным показала точность 90.8. 
# Вторая сеть - 91.8 Сеть LeNet - 5 - 93.5 
# Различия можно объяснить тем, что при добавлении дополнительных слоев (pooling слоя в первом случае, полносвязного во втором)б
# сеть может запоминать более сложные закономерности и учить более "сложные" символы. 

# На этом рисунке можно увидеть какие буквы угадываются как правильные (они более простые)

#%%

plot_diffs(third_true_dataset, 5)

#%% md

# А на этом рисунке можно увидеть символы, которые правильно были угаданы сетью LeNet, но неправильно второй сетью 
# (имеют менее выраженные признаки символов).

#%%

plot_diffs(second_diff_dataset, 5)
