import tensorflow as tf
import numpy as np


slim = tf.contrib.slim


def sync_fixed_padding(inputs_list, kernel_size, rate=1):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    rate: An integer, rate for atrous convolution.

  Returns:
    output: A tensor of size [batch, height_out, width_out, channels] with the
      input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  padded_inputs_list = []
  for i, inputs in enumerate(inputs_list):
    with tf.device('/gpu:%d' % i):
      kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
      pad_total = kernel_size_effective - 1
      pad_beg = pad_total // 2
      pad_end = pad_total - pad_beg
      padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                      [pad_beg, pad_end], [0, 0]])
    padded_inputs_list.append(padded_inputs)

  return padded_inputs_list


def sync_relu(inputs_list):
  activated_inputs_list = []
  for i, inputs in enumerate(inputs_list):
    with tf.device('/gpu:%d' % i):
      activated_inputs = tf.nn.relu(inputs)
    activated_inputs_list.append(activated_inputs)

  return activated_inputs_list


def sync_avg_pool2d(inputs_list):
  mean_list = []
  for i, inputs in enumerate(inputs_list):
    with tf.device('/gpu:%d' % i):
      means = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
    mean_list.append(means)

  return mean_list


def sync_resize_bilinear(inputs_list, size):
  resized_inputs_list = []
  for i, inputs in enumerate(inputs_list):
    with tf.device('/gpu:%d' % i):
      resized_inputs = tf.image.resize_bilinear(inputs, size, align_corners=True)
    resized_inputs_list.append(resized_inputs)

  return resized_inputs_list


def sync_concat(inputs_list_list):
  outputs_list = []
  for i, inputs_list in enumerate(inputs_list_list):
    with tf.device('/gpu:%d' % i):
      outputs = tf.concat(inputs_list, 3)
    outputs_list.append(outputs)

  return outputs_list


def sync_dropout(inputs_list, keep_prob, is_training, scope):
  outputs_list = []
  for i, inputs in enumerate(inputs_list):
    with tf.device('/gpu:%d' % i):
      outputs = slim.dropout(
          inputs,
          keep_prob=keep_prob,
          is_training=is_training,
          scope=scope)
    outputs_list.append(outputs)

  return outputs_list

def sync_conv2d(inputs_list,
                kernel_depth,
                kernel_size,
                scope,
                stride=1,
                rate=1,
                padding='SAME'):
  outputs_list = []
  for i, inputs in enumerate(inputs_list):
    with tf.device('/gpu:%d' % i):
      outputs = slim.conv2d(inputs,
                            kernel_depth,
                            kernel_size,
                            scope=scope,
                            stride=stride,
                            rate=rate,
                            padding=padding,
                            reuse=(i != 0))
    outputs_list.append(outputs)

  return outputs_list


def sync_separable_conv2d(inputs_list,
                          kernel_size,
                          depth_multiplier,
                          stride,
                          rate,
                          padding,
                          scope):
  outputs_list = []
  for i, inputs in enumerate(inputs_list):
    with tf.device('/gpu:%d' % i):
      outputs = slim.separable_conv2d(inputs,
                                      None,
                                      kernel_size,
                                      depth_multiplier=depth_multiplier,
                                      stride=stride,
                                      rate=rate,
                                      padding=padding,
                                      scope=scope,
                                      reuse=(i != 0))
    outputs_list.append(outputs)

  return outputs_list


def sync_batch_norm(xs,
                    is_training,
                    name="BatchNorm",
                    decay=0.9997,
                    epsilon=0.001):
  shape_x = xs[0].get_shape().as_list()
  dim_x = len(shape_x)
  c_x = shape_x[-1]

  with tf.variable_scope(name) as scope:
    with tf.device("/gpu:0"):
      beta = tf.get_variable(
          'beta',
          c_x,
          initializer=tf.constant_initializer(0.0),
          trainable=is_training)
      gamma = tf.get_variable(
          'gamma',
          c_x,
          initializer=tf.constant_initializer(1.0),
          trainable=is_training)
      moving_mean = tf.get_variable(
          'moving_mean',
          c_x,
          initializer=tf.constant_initializer(0.0),
          trainable=False)
      moving_var = tf.get_variable(
          'moving_variance',
          c_x,
          initializer=tf.constant_initializer(1.0),
          trainable=False)

    outputs = []
    if is_training:
      # Update moving mean and variance before applying batch normalization.
      ## step 1, collect mean & var from each mini-batch
      means = []
      vars = []
      axes = [0, 1, 2]
      for i, x in enumerate(xs):
        with tf.device('/gpu:%d' % i):
          b_mean, b_var = tf.nn.moments(xs, axes, name='moments')
          means.append(b_mean)
          vars.append(b_var)

      with tf.device("/gpu:0"):
        batch_mean = tf.add_n(means) / len(means)
        batch_var = tf.add_n(vars) / len(vars)

        # step 2, update moving mean and var operation to update variable by
        # moving average.
        update_moving_mean = tf.assign(
            moving_mean,
            moving_mean*decay + batch_mean*(1-decay))

        update_moving_var = tf.assign(
            moving_var,
            moving_var*decay + batch_var*(1-decay))
        update_ops = [update_moving_mean, update_moving_var]

      # step 3, nomarlize the batch on each GPU device.
      with tf.control_dependencies(update_ops):
        for i,x in enumerate(xs):
          with tf.device(x.device):
            output = tf.nn.batch_normalization(
                x,
                batch_mean,
                batch_var,
                beta,
                gamma,
                epsilon)
            outputs.append(output)

    else:
      for i,x in enumerate(xs):
        with tf.device(x.device):
          output = tf.nn.batch_normalization(
              x,
              moving_mean,
              moving_var,
              beta,
              gamma,
              epsilon)
          outputs.append(output)

  return outputs
