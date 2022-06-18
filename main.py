import tensorflow
import gzip
import struct
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
import tensorflow._api.v2.compat.v1 as tf
import warnings
from scipy.special import expit
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
tf.disable_v2_behavior()    # gets placeholders from older tf version

# Section 1: Implementation of Tensorflow
# One dimensional dataset; z = w*x+b
# Create graph
g = tf.Graph()
with g.as_default():
    x = tf.placeholder(dtype=tf.float32,
                       shape=(None), name='x')
    w = tf.Variable(2.0, name='weight')
    b = tf.Variable(0.7, name='bias')

    z = w*x + b
    init = tf.global_variables_initializer()

# Create session, pass in graph g
with tf.Session(graph=g) as sess:
    # Initialize w and b
    sess.run(init)
    # Evaluate z
    for t in [1.0, 0.6, -1.8]:
        print('x=%4.1f --> z=%4.1f' % (
            t, sess.run(z, feed_dict={x: t})))

with tf.Session(graph=g) as sess:
    sess.run(init)
    # print(sess.run(z, feed_dict={x: [1., 2., 3.]}))

# Working with Array Structures
g = tf.Graph()
with g.as_default():
    x = tf.placeholder(dtype=tf.float32,
                       shape=(None, 2, 3),
                       name='input_x')

    x2 = tf.reshape(x, shape=(-1, 6),
                    name='x2')

    # Calculate the sum of each column
    xsum = tf.reduce_sum(x2, axis=0, name='col_sum')

    # Calculate the mean of each column
    xmean = tf.reduce_mean(x2, axis=0, name='col_mean')

with tf.Session(graph=g) as sess:
    x_array = np.arange(18).reshape(3, 2, 3)

    # print('Input Shape: ', x_array.shape)
    # print('Reshaped: \n',
    #       sess.run(x2, feed_dict={x: x_array}))
    # print('Column Sums:\n',
    #       sess.run(xsum, feed_dict={x: x_array}))
    # print('Column Means:\n',
    #       sess.run(xmean, feed_dict={x: x_array}))



# Section 2: Developing Simple Model with Low-Level TF API
X_train = np.arange(10).reshape((10, 1))
y_train = np.array([1.0, 1.3, 3.1,
                    2.0, 5.0, 6.3,
                    6.6, 7.4, 8.0,
                    9.0])

# Want to train a linear regression model to predict the output y from the input x
# define linreg model as z = w*x + b; define cost fxn as MSE; define variables; use gradient descent optimizer
class TfLinreg(object):

    def __init__(self, x_dim, learning_rate=0.01,
                 random_seed=None):
        self.x_dim = x_dim
        self.learning_rate = learning_rate
        self.g = tf.Graph()
        # Build model
        with self.g.as_default():
            # Set graph-level random-seed
            tf.set_random_seed(random_seed)

            self.build()
            # Create initializer
            self.init_op = tf.global_variables_initializer()

    def build(self):
        # Define placeholders for inputs
        self.X = tf.placeholder(dtype=tf.float32,
                                shape=(None, self.x_dim),
                                name='x_input')
        self.y = tf.placeholder(dtype=tf.float32,
                                shape=(None),
                                name='y_input')
        # print(self.X)
        # print(self.y)
        # Define weight matrix and bias vector
        w = tf.Variable(tf.zeros(shape=(1)),
                        name='weight')
        b = tf.Variable(tf.zeros(shape=(1)),
                        name='bias')
        # print(w)
        # print(b)

        self.z_net = tf.squeeze(w*self.X + b,
                                name='z_net')
        # print(self.z_net)

        sqr_errors = tf.square(self.y - self.z_net,
                                name='sqr_errors')
        # print(sqr_errors)
        self.mean_cost = tf.reduce_mean(sqr_errors,
                                        name='mean_cost')

        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate,
            name='GradientDescent')
        self.optimizer = optimizer.minimize(self.mean_cost)

# Create instance
lrmodel = TfLinreg(x_dim=X_train.shape[1], learning_rate=0.01)

# Implement training fxn to learn weights of the linear regression model
def train_linreg(sess, model, X_train, y_train, num_epochs=10):
    # Initialize all variables: W and b
    sess.run(model.init_op)

    training_costs = []
    for i in range(num_epochs):
        _, cost = sess.run([model.optimizer, model.mean_cost],
                           feed_dict={model.X: X_train,
                                      model.y: y_train})
        training_costs.append(cost)

    return training_costs

# Create new TF session; launch lrmodel.g graph and pass required arguments to train_linreg for training
sess = tf.Session(graph=lrmodel.g)
training_costs = train_linreg(sess, lrmodel, X_train, y_train)

# Visualize training costs after 10 epochs; look to see if model converged or not
plt.plot(range(1, len(training_costs) + 1), training_costs)
plt.tight_layout()
plt.xlabel('Epoch')
plt.ylabel('Training Cost')
plt.show()  # converges very quickly after a few epochs

# Compile new function to make predictions base don input features
def predict_linreg(sess, model, X_test):
    y_pred = sess.run(model.z_net,
                      feed_dict={model.X: X_test})
    return y_pred

plt.scatter(X_train, y_train,
            marker='s', s=50,
            label='Training Data')
plt.plot(range(X_train.shape[0]),
               predict_linreg(sess, lrmodel, X_train),
               color='gray', marker='o',
               markersize=6, linewidth=3,
               label='LinReg Model')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
plt.show()  # model fits training data points



# Section 2: Training NNs efficiently w/ high-level TF APIs
# Building Multilayer NN's using TF's Layers API
# Load data
def load_mnist(path, kind='train'):
    """Load MNIST Data from 'path'"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(
            len(labels), 784)
        images = ((images / 255.) - 0.5) * 2

    return images, labels

# Unzip mnist
if sys.version_info > (3, 0):
    writemode = 'wb'
else:
    writemode = 'w'

zipped_mnist = [f for f in os.listdir('./samples') if f.endswith('ubyte.gz')]
for z in zipped_mnist:
    with gzip.GzipFile(z, mode='rb') as decompressed, open(z[:-3], writemode) as outfile:
        outfile.write(decompressed.read())

X_train, y_train = load_mnist('./samples', kind='train')
print('Rows: %d, Columns: %d' % (X_train.shape[0],
                                 X_train.shape[1]))

X_test, y_test = load_mnist('./samples', kind='t10k')
print('Rows: %d, Columns: %d' % (X_test.shape[0],
                                 X_test.shape[1]))
# Mean centering and normalization
mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)

X_train_centered = (X_train - mean_vals)/std_val
X_test_centered = (X_test - mean_vals)/std_val

del X_train, X_test

print(X_train_centered.shape, y_train.shape)
print(X_test_centered.shape, y_test.shape)


# Build Model; multilayer perceptron w/ 3 fully connected layers
# Create placeholders tf_x, tf_y
n_features = X_train_centered.shape[1]
n_classes = 10
random_seed = 123
np.random.seed(random_seed)

g = tf.Graph()
with g.as_default():
    tf.set_random_seed(random_seed)
    tf_x = tf.placeholder(dtype=tf.float32,
                          shape=(None, n_features),
                          name='tf_x')

    tf_y = tf.placeholder(dtype=tf.int32,
                          shape=None, name='tf_y')
    y_onehot = tf.one_hot(indices=tf_y, depth=n_classes)

    h1 = tf.layers.dense(inputs=tf_x, units=50,
                         activation=tf.tanh,
                         name='layer1')

    h2 = tf.layers.dense(inputs=h1, units=50,
                         activation=tf.tanh,
                         name='layer2')

    logits = tf.layers.dense(inputs=h2,
                             units=10,
                             activation=None,
                             name='layer3')

    predictions = {
        'classes': tf.argmax(logits, axis=1,
                             name='predicted_classes'),
        'probabilities': tf.nn.softmax(logits,
                                       name='softmax_tensor')
    }

# Define cost fxn and optimizer
with g.as_default():
    cost = tf.losses.softmax_cross_entropy(
        onehot_labels=y_onehot, logits=logits)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

    train_op = optimizer.minimize(loss=cost)

    init_op = tf.global_variables_initializer()

# fxn to generate batches of data
def create_batch_generator(X, y, batch_size=128, shuffle=False):
    X_copy = np.array(X)
    y_copy = np.array(y)

    if shuffle:
        data = np.column_stack((X_copy, y_copy))
        np.random.shuffle(data)
        X_copy = data[:, :-1]
        y_copy = data[:, -1].astype(int)

    for i in range(0, X.shape[0], batch_size):
        yield X_copy[i:i + batch_size, :], y_copy[i:i + batch_size]

# Create session to launch graph
sess = tf.Session(graph=g)
# Run the variable initialization operator
sess.run(init_op)

# 50 epochs of training
for epoch in range(50):
    training_costs = []
    batch_generator = create_batch_generator(
        X_train_centered, y_train,
        batch_size=64)
    for batch_X, batch_y in batch_generator:
        # Prepare a dict to feed data to network
        feed = {tf_x: batch_X, tf_y: batch_y}
        _, batch_cost = sess.run([train_op, cost], feed_dict=feed)
        training_costs.append(batch_cost)
    print(' -- Epoch %2d  '
          'Avg. Training Loss: %.4f' % (
              epoch+1, np.mean(training_costs)
    ))

# make prediction on the test set
feed = {tf_x: X_test_centered}
y_pred = sess.run(predictions['classes'],
                  feed_dict=feed)

print('Test Accuracy: %.2f%%' % (
      100*np.sum(y_pred == y_test)/y_test.shape[0]))



# Section 3: Developing Multilayer NN w/ Keras
# Load data
X_train, y_train = load_mnist('./samples', kind='train')
print('Rows: %d, Columns: %d' % (X_train.shape[0],
                                 X_train.shape[1]))

X_test, y_test = load_mnist('./samples', kind='t10k')
print('Rows: %d, Columns: %d' % (X_test.shape[0],
                                 X_test.shape[1]))

# Mean centering and normalization
mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)

X_train_centered = (X_train - mean_vals)/std_val
X_test_centered = (X_test - mean_vals)/std_val

del X_train, X_test

print(X_train_centered.shape, y_train.shape)
print(X_test_centered.shape, y_test.shape)


# Set random seed for NP and TF to get consistent results
np.random.seed(123)
tf.set_random_seed(123)

# Convert class labels into one_hot format using keras
y_train_onehot = keras.utils.to_categorical(y_train)
print('First 3 labels: ', y_train[:3])
print('\nFirst 3 labels (one-hot):\n', y_train_onehot[:3])

# Implement Neural Network
# First 2 layers have 50 hidden units w/ tanh activation fxn
# Last layer has 10 layers for 10 class labels, uses softmax to give probability of each class
model = keras.models.Sequential()

model.add(
    keras.layers.Dense(
        units=50,
        input_dim=X_train_centered.shape[1],
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='tanh'))

model.add(
    keras.layers.Dense(
        units=50,
        input_dim=50,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='tanh'))

model.add(
    keras.layers.Dense(
        units=y_train_onehot.shape[1],
        input_dim=50,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='softmax'))

sgd_optimizer = keras.optimizers.SGD(
        lr=0.001, decay=1e-7, momentum=.9)

model.compile(optimizer=sgd_optimizer,
              loss='categorical_crossentropy')

# Train
history = model.fit(X_train_centered, y_train_onehot,
                    batch_size=64, epochs=50,
                    verbose=1,
                    validation_split=0.1)

# Predict class labels
y_train_pred = np.argmax(model.predict(X_train_centered, verbose=0), axis=1)
print('First 3 predictions: ', y_train_pred[:3])

# Print model accuracy on training and test sets:
y_train_pred = np.argmax(model.predict(X_train_centered, verbose=0), axis=1)
correct_preds = np.sum(y_train == y_train_pred, axis=0)
train_acc = correct_preds / y_train.shape[0]


print('First 3 predictions: ', y_train_pred[:3])
print('Training accuracy: %.2f%%' % (train_acc * 100))

y_test_pred = np.argmax(model.predict(X_test_centered,
                                      verbose=0), axis=1)
correct_preds = np.sum(y_test == y_test_pred, axis=0)
test_acc = correct_preds / y_test.shape[0]
print('Test accuracy: %.2f%%' % (test_acc * 100))