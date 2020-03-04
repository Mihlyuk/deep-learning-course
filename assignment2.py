import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from six.moves import cPickle as pickle
from six.moves import range
import matplotlib.pyplot as plt

#### 1. Реализуйте полносвязную нейронную сеть с помощью библиотеки Tensor Flow.
# В качестве алгоритма оптимизации можно использовать, например, стохастический градиент (Stochastic Gradient Descent, SGD).
# Определите количество скрытых слоев от 1 до 5, количество нейронов в каждом из слоев до нескольких сотен,
# а также их функции активации (кусочно-линейная, сигмоидная, гиперболический тангенс и т.д.).

pickle_file = 'data/notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']

    del save  # for gc

    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
    return np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0] * 100.0


batch_size = 128
num_steps = 5000
hidden_size1 = 300
hidden_size2 = 300
hidden_size3 = 300
beta_val = np.logspace(-4, -2, 20)
accuracy_val = []

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables
    W1 = tf.get_variable('W1', [image_size * image_size, hidden_size1], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable('b1', [hidden_size1], initializer=tf.zeros_initializer())

    # Hidden layer 1
    W2 = tf.get_variable('W2', [hidden_size1, hidden_size2], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable('b2', [hidden_size2], initializer=tf.zeros_initializer())

    # Hidden layer 2
    W3 = tf.get_variable('W3', [hidden_size2, hidden_size3], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable('b3', [hidden_size3], initializer=tf.zeros_initializer())

    # Hidden layer 3
    W4 = tf.get_variable('W4', [hidden_size3, num_labels], initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable('b4', [num_labels], initializer=tf.zeros_initializer())

    # Training computation
    a1 = tf.nn.relu(tf.matmul(tf_train_dataset, W1) + b1)
    a2 = tf.nn.relu(tf.matmul(a1, W2) + b2)
    a3 = tf.nn.relu(tf.matmul(a2, W3) + b3)
    logits = tf.matmul(a3, W4) + b4

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    train_prediction = tf.nn.softmax(logits)

    a1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, W1) + b1)
    a2_valid = tf.nn.relu(tf.matmul(a1_valid, W2) + b2)
    a3_valid = tf.nn.relu(tf.matmul(a2_valid, W3) + b3)
    valid_logits = tf.matmul(a3_valid, W4) + b4
    valid_prediction = tf.nn.softmax(valid_logits)

    a1_test = tf.nn.relu(tf.matmul(tf_test_dataset, W1) + b1)
    a2_test = tf.nn.relu(tf.matmul(a1_test, W2) + b2)
    a3_test = tf.nn.relu(tf.matmul(a2_test, W3) + b3)
    test_logits = tf.matmul(a3_test, W4) + b4
    test_prediction = tf.nn.softmax(test_logits)

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()

    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}

        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

        if step % 500 == 0:
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(
                valid_prediction.eval(), valid_labels))

    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

#### Задание 2. Как улучшилась точность классификатора по сравнению с логистической регрессией?

# Я использовал полносвязную нейронную сеть с тремя слоями по 300 нейронов в каждой,
# с функцией активации RelU на скрытых слоях и softmax на выходном слое.
# Точность классификатора выросла на 5.2 процента по сравнению с логистической регрессией:
# теперь она составляет 94.8%

#### Задание 3. Используйте регуляризацию и метод сброса нейронов (dropout) для борьбы с переобучением. Как улучшилось качество классификации?

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    regularization = tf.placeholder(tf.float32)

    # Variables
    W1 = tf.get_variable('W1', [image_size * image_size, hidden_size1],
                         initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable('b1', [hidden_size1], initializer=tf.zeros_initializer())

    # Hidden layer 1
    W2 = tf.get_variable('W2', [hidden_size1, hidden_size2], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable('b2', [hidden_size2], initializer=tf.zeros_initializer())

    # Hidden layer 2
    W3 = tf.get_variable('W3', [hidden_size2, hidden_size3], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable('b3', [hidden_size3], initializer=tf.zeros_initializer())

    # Hidden layer 3
    W4 = tf.get_variable('W4', [hidden_size3, num_labels], initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable('b4', [num_labels], initializer=tf.zeros_initializer())

    # Training computation
    a1 = tf.nn.relu(tf.matmul(tf_train_dataset, W1) + b1)
    a2 = tf.nn.relu(tf.matmul(a1, W2) + b2)
    a3 = tf.nn.relu(tf.matmul(a2, W3) + b3)
    logits = tf.matmul(a3, W4) + b4

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels)) + \
           regularization * (
                   tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1) +
                   tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2) +
                   tf.nn.l2_loss(W3) + tf.nn.l2_loss(b3) +
                   tf.nn.l2_loss(W4) + tf.nn.l2_loss(b4))

    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    train_prediction = tf.nn.softmax(logits)

    a1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, W1) + b1)
    a2_valid = tf.nn.relu(tf.matmul(a1_valid, W2) + b2)
    a3_valid = tf.nn.relu(tf.matmul(a2_valid, W3) + b3)
    valid_logits = tf.matmul(a3_valid, W4) + b4
    valid_prediction = tf.nn.softmax(valid_logits)

    a1_test = tf.nn.relu(tf.matmul(tf_test_dataset, W1) + b1)
    a2_test = tf.nn.relu(tf.matmul(a1_test, W2) + b2)
    a3_test = tf.nn.relu(tf.matmul(a2_test, W3) + b3)
    test_logits = tf.matmul(a3_test, W4) + b4
    test_prediction = tf.nn.softmax(test_logits)


for beta in beta_val:
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()

        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, regularization: beta}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

        print("L2 regularization(beta=%.5f) Test accuracy: %.1f%%" % (beta, accuracy(test_prediction.eval(), test_labels)))
        accuracy_val.append(accuracy(test_prediction.eval(), test_labels))

print('Best beta=%f, accuracy=%.1f%%' % (beta_val[np.argmax(accuracy_val)], max(accuracy_val)))
plt.semilogx(beta_val, accuracy_val)
plt.grid(True)
plt.title('Test accuracy by regularization (logistic)')
plt.show()

# Для данной НС регуляризация почти не улучшила точность классификатора (+0.1% в сравнении с предыдущей моделью).
# Далее будем использовать значении регуляризации 0.00016, при котором НС показала наилучшую точность 94.9%

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    regularization = tf.placeholder(tf.float32)

    # Variables
    W1 = tf.get_variable('W1', [image_size * image_size, hidden_size1],
                         initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable('b1', [hidden_size1], initializer=tf.zeros_initializer())

    # Hidden layer 1
    W2 = tf.get_variable('W2', [hidden_size1, hidden_size2], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable('b2', [hidden_size2], initializer=tf.zeros_initializer())

    # Hidden layer 2
    W3 = tf.get_variable('W3', [hidden_size2, hidden_size3], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable('b3', [hidden_size3], initializer=tf.zeros_initializer())

    # Hidden layer 3
    W4 = tf.get_variable('W4', [hidden_size3, num_labels], initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable('b4', [num_labels], initializer=tf.zeros_initializer())

    # Training computation
    a1 = tf.nn.relu(tf.matmul(tf_train_dataset, W1) + b1)
    a1 = tf.nn.dropout(a1, 0.9)
    a2 = tf.nn.relu(tf.matmul(a1, W2) + b2)
    a2 = tf.nn.dropout(a2, 0.9)
    a3 = tf.nn.relu(tf.matmul(a2, W3) + b3)
    a3 = tf.nn.dropout(a3, 0.9)
    logits = tf.matmul(a3, W4) + b4

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels)) + \
           regularization * (
                   tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1) +
                   tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2) +
                   tf.nn.l2_loss(W3) + tf.nn.l2_loss(b3) +
                   tf.nn.l2_loss(W4) + tf.nn.l2_loss(b4))

    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    train_prediction = tf.nn.softmax(logits)

    a1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, W1) + b1)
    a2_valid = tf.nn.relu(tf.matmul(a1_valid, W2) + b2)
    a3_valid = tf.nn.relu(tf.matmul(a2_valid, W3) + b3)
    valid_logits = tf.matmul(a3_valid, W4) + b4
    valid_prediction = tf.nn.softmax(valid_logits)

    a1_test = tf.nn.relu(tf.matmul(tf_test_dataset, W1) + b1)
    a2_test = tf.nn.relu(tf.matmul(a1_test, W2) + b2)
    a3_test = tf.nn.relu(tf.matmul(a2_test, W3) + b3)
    test_logits = tf.matmul(a3_test, W4) + b4
    test_prediction = tf.nn.softmax(test_logits)

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()

    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, regularization: 0.00016}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

        if step % 500 == 0:
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))

    print("Dropout Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

# Применение техники dropout не привело к увеличению качества классфикации.
# Вероятно НС не имеет признаков переобучения и поэтому, в данном случае, применение dropout только увеличивает ошибку обучения.

#### Задание 4. Воспользуйтесь динамически изменяемой скоростью обучения (learning rate). Наилучшая точность, достигнутая с помощью данной модели составляет 97.1%. Какую точность демонстрирует Ваша реализованная модель?

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    regularization = tf.placeholder(tf.float32)
    global_step = tf.Variable(0)

    # Variables
    W1 = tf.Variable(
        tf.truncated_normal([image_size * image_size, hidden_size1], stddev=np.sqrt(2.0 / (image_size * image_size))))
    b1 = tf.Variable(tf.zeros([hidden_size1]))

    W2 = tf.Variable(tf.truncated_normal([hidden_size1, hidden_size2], stddev=np.sqrt(2.0 / hidden_size1)))
    b2 = tf.Variable(tf.zeros([hidden_size2]))

    W3 = tf.Variable(tf.truncated_normal([hidden_size2, hidden_size3], stddev=np.sqrt(2.0 / hidden_size2)))
    b3 = tf.Variable(tf.zeros([hidden_size3]))

    W4 = tf.Variable(tf.truncated_normal([hidden_size3, num_labels], stddev=np.sqrt(2.0 / hidden_size3)))
    b4 = tf.Variable(tf.zeros([num_labels]))

    # Training computation
    a1 = tf.nn.relu(tf.matmul(tf_train_dataset, W1) + b1)
    a2 = tf.nn.relu(tf.matmul(a1, W2) + b2)
    a3 = tf.nn.relu(tf.matmul(a2, W3) + b3)
    logits = tf.matmul(a3, W4) + b4

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels)) + \
           regularization * (
                   tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1) +
                   tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2) +
                   tf.nn.l2_loss(W3) + tf.nn.l2_loss(b3) +
                   tf.nn.l2_loss(W4) + tf.nn.l2_loss(b4))

    learning_rate = tf.train.exponential_decay(0.5, global_step, 1000, 0.7, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    train_prediction = tf.nn.softmax(logits)

    a1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, W1) + b1)
    a2_valid = tf.nn.relu(tf.matmul(a1_valid, W2) + b2)
    a3_valid = tf.nn.relu(tf.matmul(a2_valid, W3) + b3)
    valid_logits = tf.matmul(a3_valid, W4) + b4
    valid_prediction = tf.nn.softmax(valid_logits)

    a1_test = tf.nn.relu(tf.matmul(tf_test_dataset, W1) + b1)
    a2_test = tf.nn.relu(tf.matmul(a1_test, W2) + b2)
    a3_test = tf.nn.relu(tf.matmul(a2_test, W3) + b3)
    test_logits = tf.matmul(a3_test, W4) + b4
    test_prediction = tf.nn.softmax(test_logits)

num_steps = 5000
hidden_size1 = 4000
hidden_size2 = 2000
hidden_size3 = 100

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()

    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, regularization: 0.00016}

        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

        if step % 500 == 0:
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))

    print("Final Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


# С помощью динамически изменяемой скорости обучения удалось достичь точности 95.3%
