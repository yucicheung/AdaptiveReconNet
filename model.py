from utils import (
  read_data, 
  input_setup, 
  imsave,
  merge
)

import time
import os

import numpy as np
import tensorflow as tf

class RECONNET(object):

  def __init__(self, 
               sess, 
               image_size=33,
               label_size=33,
               batch_size=128,
               c_dim=1, 
               checkpoint_dir=None, 
               sample_dir=None):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_size = image_size
    self.label_size = label_size
    self.batch_size = batch_size

    self.c_dim = c_dim

    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.build_model()

  def build_model(self):
    self.images = tf.placeholder(tf.float32, [None, self.image_size,self.image_size,  self.c_dim], name='images')
    self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels')


    self.weights = {
      'fc1w': tf.Variable(tf.random_normal([1089,109], stddev=1e-2), name='fc1w'),
      'fc2w': tf.Variable(tf.random_normal([109,1089], stddev=1e-2), name='fc2w'),
      'w1': tf.Variable(tf.random_normal([11, 11, 1, 64], stddev=1e-1), name='w1'),
      'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-1), name='w2'),
      'w3': tf.Variable(tf.random_normal([7, 7, 32, 1], stddev=1e-1), name='w3'),
      'w4': tf.Variable(tf.random_normal([11, 11, 1, 64], stddev=1e-1), name='w4'),
      'w5': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-1), name='w5'),
      'w6': tf.Variable(tf.random_normal([7, 7, 32, 1], stddev=1e-1), name='w6'),

    }
    self.biases = {
      'fc1b': tf.Variable(tf.zeros([109]), name='fc1b'),
      'fc2b': tf.Variable(tf.zeros([1089]), name='fc2b'),
      'b1': tf.Variable(tf.zeros([64]), name='b1'),
      'b2': tf.Variable(tf.zeros([32]), name='b2'),
      'b3': tf.Variable(tf.zeros([1]), name='b3'),
      'b4': tf.Variable(tf.zeros([64]), name='b4'),
      'b5': tf.Variable(tf.zeros([32]), name='b5'),
      'b6': tf.Variable(tf.zeros([1]), name='b6')

    }

    self.pred = self.model()

    # Loss function (MSE)
    self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))

    self.saver = tf.train.Saver()

  def train(self, config):
    if config.is_train:
      input_setup(self.sess, config)
    else:
      nx, ny, pad_h, pad_w = input_setup(self.sess, config)

    if config.is_train:
      data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")
    else:
      data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")

    train_data, train_label = read_data(data_dir)

    # Stochastic gradient descent
    self.train_op = tf.train.MomentumOptimizer(config.learning_rate,0.9).minimize(self.loss)

    tf.global_variables_initializer().run()
    
    counter = 0
    start_time = time.time()

    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    if config.is_train:
      print("Training...")

      for ep in xrange(config.epoch):
        # Run by batch images
        batch_idxs = len(train_data) // config.batch_size
        for idx in xrange(0, batch_idxs):
          batch_images = train_data[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels = train_label[idx*config.batch_size : (idx+1)*config.batch_size]

          counter += 1
          _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})

          if counter % 10 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
              % ((ep+1), counter, time.time()-start_time, err))

          if counter % 500 == 0:
            self.save(config.checkpoint_dir, counter)

    else:
      print("Testing...")

      result = self.pred.eval({self.images: train_data, self.labels: train_label})

      result = merge(result, [nx, ny])
      result = result.squeeze()

      # change back to original size
      h, w = np.shape(result)
      result = result[0:(h-pad_h), 0:(w-pad_w)]
      image_path = os.path.join(os.getcwd(), config.sample_dir)
      image_path = os.path.join(image_path, "test.png")
      imsave(result, image_path)

  def model(self):
    flattenimg = tf.reshape(self.images,[-1,self.image_size * self.image_size * self.c_dim])
    fc1 = tf.matmul(flattenimg,self.weights['fc1w']) + self.biases['fc1b']
    fc2 = tf.matmul(fc1,self.weights['fc2w']) + self.biases['fc2b']
    fc2_reshape = tf.reshape(fc2,[-1,self.image_size,self.image_size,  self.c_dim])
    conv1 = tf.nn.relu(tf.nn.conv2d(fc2_reshape, self.weights['w1'], strides=[1,1,1,1], padding='SAME') + self.biases['b1'])
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, self.weights['w2'], strides=[1,1,1,1], padding='SAME') + self.biases['b2'])
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, self.weights['w3'], strides=[1,1,1,1], padding='SAME') + self.biases['b3'])
    conv4 = tf.nn.relu(tf.nn.conv2d(conv3, self.weights['w4'], strides=[1, 1, 1, 1], padding='SAME') + self.biases['b4'])
    conv5 = tf.nn.relu(tf.nn.conv2d(conv4, self.weights['w5'], strides=[1, 1, 1, 1], padding='SAME') + self.biases['b5'])
    conv6 = tf.nn.conv2d(conv5, self.weights['w6'], strides=[1, 1, 1, 1], padding='SAME') + self.biases['b6']
    return conv6

  def save(self, checkpoint_dir, step):
    model_name = "Reconnet.model"
    model_dir = "%s_%s" % ("reconnet", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("reconnet", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False
