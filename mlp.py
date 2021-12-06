import tensorflow as tf
# from img_classification import load_data_fashion_mnist
import img_classification
# import matplotlib.pyplot as plt

# def prelu(x):
#   return tf.maximum(0, x) + 0.1 * tf.minimum(0, x)

# x = tf.Variable(tf.range(-8.0, 8.0, 0.1), dtype=tf.float32)
# # y = tf.nn.relu(x)
# # y = prelu(x)
# # y = tf.nn.sigmoid(x)
# # y = tf.nn.tanh(x)
# # plt.plot(x.numpy(), y.numpy())
# # plt.show()


# with tf.GradientTape() as t:
#   y = tf.nn.sigmoid(x)

# plt.plot(x.numpy(), t.gradient(y, x).numpy(), label='sigmoid')

# with tf.GradientTape() as t:
#   y = tf.nn.tanh(x)
# plt.plot(x.numpy(), t.gradient(y, x).numpy(), label='tanh')

# plt.show()


batch_size = 256
train_iter, test_iter = img_classification.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = tf.Variable(tf.random.normal(shape=(num_inputs, num_hiddens), mean=0, stddev=0.01))
b1 = tf.Variable(tf.zeros(num_hiddens))
W2 = tf.Variable(tf.random.normal(num_hiddens, num_outputs), mean=0, stddev=0.01)
b2 = tf.Variable(tf.random.normal([num_outputs], stddev=0.01))

params = [W1, b1, W2, b2]

def relu(X):
  """Activation function."""
  return tf.math.maximum(X, 0)

def net(X):
  """Model"""
  X = tf.reshape(X, (-1, num_inputs))
  H = relu(tf.matmul(X, W1) + b1)
  return tf.matmul(H, W2) + b2

def loss(y_hat, y):
  return tf.losses.sparse_categorical_crossentropy(y, y_hat, from_logits=True)

num_epochs, lr = 10, 0.1
updater = img_classification.Updater([W1, W2, b1, b2], lr)
img_classification.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)