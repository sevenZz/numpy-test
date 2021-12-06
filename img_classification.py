import tensorflow as tf
import matplotlib.pyplot as plt


def load_data_fashion_mnist(batch_size, resize=None):
  """Download the Fashion_MNIST dataset and load it into memory."""
  mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()

  process = lambda X, y: (tf.expand_dims(X, axis=3) / 255, tf.cast(y, dtype='int32'))
  resize_fn = lambda X, y: (tf.image.resize_with_pad(X, resize, resize) if resize else X, y)
  return (tf.data.Dataset.from_tensor_slices(process(*mnist_train)).batch(batch_size).shuffle(len(mnist_train[0])).map(resize_fn),
          tf.data.Dataset.from_tensor_slices(process(*mnist_test)).batch(batch_size).map(resize_fn))
  

def get_fashion_mnist_labels(labels):
  """Return text labels for the Fashion-MNIST dataset."""
  text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
  return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
  """Plot a list of images."""
  figsize = (num_cols * scale, num_rows * scale)
  _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
  axes = axes.flatten()
  for i, (ax, img) in enumerate(zip(axes, imgs)):
    ax.imshow(img.numpy())
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    if titles:
      ax.set_title(titles[i])
  return axes

def soft_max(X):
  """soft max algorithm"""
  X_exp = tf.exp(X)
  partition = tf.reduce_sum(X_exp, 1, keepdims=True)
  return X_exp / partition

def net(X):
  """regression model"""
  return soft_max(tf.matmul(tf.reshape(X, (-1, W.shape[0])), W) + b)

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

W = tf.Variable(tf.random.normal(shape=(num_inputs, num_outputs), mean=0, stddev=0.01))
b = tf.Variable(tf.zeros(num_outputs))


y_hat = tf.constant([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = tf.constant([0, 2])
# tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))

def cross_entropy(y_hat, y):
  return -tf.math.log(tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1])))

def accuracy(y_hat ,y):
  """Compute the number of correct predictions"""
  if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
    y_hat = tf.argmax(y_hat, axis=1)
  cmp = tf.cast(y_hat, y.dtype) == y
  return float(tf.reduce_sum(tf.cast(cmp, y.dtype)))

class Accumulator:
  """For accumulating sums over `n` variables."""
  def __init__(self, n) -> None:
      self.data = [0.0] * n

  def add(self, *args):
    self.data = [a + float(b) for a, b in zip(self.data, args)]

  def reset(self):
    self.data = [0.0] * len(self.data)
  
  def __getitem__(self, idx):
    return self.data[idx]

def evaluate_accuracy(net, data_iter):
  metric = Accumulator(2)
  for X, y in data_iter:
    metric.add(accuracy(net(X), y), tf.size(y).numpy())
  return metric[0] / metric[1]

def train_epoch_ch3(net, train_iter, loss, updater):
  metric = Accumulator(3)
  for X, y in train_iter:
    with tf.GradientTape() as tape:
      y_hat = net(X)
      if isinstance(loss, tf.keras.losses.Loss):
        l = loss(y, y_hat)
      else:
        l = loss(y_hat, y)

      if isinstance(updater, tf.keras.optimizers.Optimizer):
        params = net.trainable_variables
        grads = tape.gradient(l, params)
        updater.apply_gradients(zip(grads, params))
      else:
        updater(X.shape[0], tape.gradient(l, updater.params))
      
      l_sum = l * float(tf.size(y)) if isinstance(loss, tf.keras.losses.Loss) else tf.reduce_sum(l)
      metric.add(l_sum, accuracy(y_hat, y), tf.size(y))
    return metric[0]/ metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
  for epoch in range(num_epochs):
    train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
    test_acc = evaluate_accuracy(net, test_iter)
  train_loss, train_acc = train_metrics
  assert train_loss < 0.5, train_loss
  assert train_acc <= 1 and train_acc > 0.7, train_acc
  assert test_acc <= 1 and test_acc > 0.7, test_acc

class Updater():
  def __init__(self, params, lr):
      self.params = params
      self.lr = lr

  def __call__(self, batch_size, grads):
    self.sgd(self.params, grads, self.lr, batch_size)

  def sgd(self, params, grads, lr, batch_size):
    for param, grad in zip(params, grads):
      param.assign_sub(lr * grad / batch_size)

updater = Updater([W, b], lr=0.1)
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)