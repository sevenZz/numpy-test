import numpy as np
import tensorflow as tf
import random
import pandas as pd

# w = tf.Variable(tf.random.normal(shape=(2, 1), mean=0, stddev=0.01), trainable=True)
# b = tf.Variable(tf.zeros(1), trainable=True)

# print(w)

# print(b)
# data = pd.read_excel('/Users/zhangzhao/Downloads/ENB2012_data.xlsx')
# data = data.corr()
# data = data.iloc[:len(data)-2, :len(data)-2]
# print(data)


def synthetic_data(w, b, num_examples):
  """Generate  y = Xw + b + noises"""
  X = tf.zeros((num_examples, w.shape[0]))
  X += tf.random.normal(shape=X.shape)
  y = tf.matmul(X, tf.reshape(w, (-1, 1))) + b
  y += tf.random.normal(shape=y.shape, stddev=0.01)
  y = tf.reshape(y, (-1, 1))
  return X, y

true_w = tf.constant([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print('features:', features[0], '\nlabel:', labels[0])

# def data_iter(batch_size, features, labels):
#   n = len(features)
#   indices = list(range(n))
#   random.shuffle(indices)
  
#   for i in range(0, n, batch_size):
#     j = tf.constant(indices[i : min(i + batch_size, n)])
#     yield tf.gather(features, j), tf.gather(labels, j)

# batch_size = 10

# for X, y in data_iter(batch_size, features, labels):
#   print(X, '\n', y)
#   break

def load_array(data_arrays, batch_size, is_train=True):
  dataset = tf.data.Dataset.from_tensor_slices(data_arrays)
  if is_train:
    dataset = dataset.shuffle(buffer_size=1000)
  dataset = dataset.batch(batch_size)
  return dataset

batch_size = 10
data_iter = load_array((features, labels), batch_size)

print(next(iter(data_iter)))

initializer = tf.initializers.RandomNormal(stddev=0.01)

net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1, kernel_initializer=initializer))

# Loss Function
loss = tf.keras.losses.MeanSquaredError()

# Optimization Algorithm
trainer = tf.keras.optimizers.SGD(learning_rate=0.03)

# Train
num_epochs = 3
for epoch in range(num_epochs):
  for X, y in data_iter:
    with tf.GradientTape() as tape:
      l = loss(net(X, training=True), y)
    grads = tape.gradient(l, net.trainable_variables)
    trainer.apply_gradients(zip(grads, net.trainable_variables))
  l = loss(net(features), labels)
  print(f'epoch {epoch + 1}, loss {l:f}')

w = net.get_weights()[0]
b = net.get_weights()[1]
print(f'w: {w}, b: {b}')

# w = tf.Variable(tf.random.normal(shape=(2,1), mean=0, stddev=0.01), trainable=True)
# b = tf.Variable(tf.zeros(1), trainable=True)

# print(w, '\n', b)

# def linreg(X, w, b):
#   return tf.matmul(X, w) + b

# def loss_func(y, y_hat):
#   return (y_hat - tf.reshape(y, y_hat.shape)) ** 2 / 2

# def sgd(params, grads, lr, batch_size):
#   for param, grad in zip(params, grads):
#     param.assign_sub(lr * grad / batch_size)

# lr = 0.03
# num_epochs = 3
# net = linreg
# loss = loss_func

# for epoch in range(num_epochs):
#   for X, y in data_iter(batch_size, features, labels):
#     with tf.GradientTape() as g:
#       l = loss(y, net(X, w, b))
#     dw, db = g.gradient(l, [w, b])
#     sgd([w, b], [dw, db], lr, batch_size)
#   train_l = loss(net(features, w, b), labels)
#   print(f'epoch {epoch + 1}, loss {float(tf.reduce_mean(train_l)):f}')