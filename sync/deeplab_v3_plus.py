import tensorflow as tf

from deeplab.sync import xception_sync
from deeplab.sync import sync_util

slim = tf.contrib.slim

sync_fixed_padding = sync_util.sync_fixed_padding
sync_relu = sync_util.sync_relu
sync_conv2d = sync_util.sync_conv2d
sync_separable_conv2d = sync_util.sync_separable_conv2d
sync_batch_norm = sync_util.sync_batch_norm
sync_avg_pool2d = sync_util.sync_avg_pool2d
sync_resize_bilinear = sync_util.sync_resize_bilinear
sync_concat = sync_util.sync_concat
sync_dropout = sync_util.sync_dropout

LOGITS_SCOPE_NAME = 'logits'
MERGED_LOGITS_SCOPE = 'merged_logits'
IMAGE_POOLING_SCOPE = 'image_pooling'
ASPP_SCOPE = 'aspp'
CONCAT_PROJECTION_SCOPE = 'concat_projection'
DECODER_SCOPE = 'decoder'
META_ARCHITECTURE_SCOPE = 'meta_architecture'


def get_extra_layer_scopes(last_layers_contain_logits_only=False):
  """Gets the scopes for extra layers.

  Args:
    last_layers_contain_logits_only: Boolean, True if only consider logits as
    the last layer (i.e., exclude ASPP module, decoder module and so on)

  Returns:
    A list of scopes for extra layers.
  """
  if last_layers_contain_logits_only:
    return [LOGITS_SCOPE_NAME]
  else:
    return [
        LOGITS_SCOPE_NAME,
        IMAGE_POOLING_SCOPE,
        ASPP_SCOPE,
        CONCAT_PROJECTION_SCOPE,
        DECODER_SCOPE,
        META_ARCHITECTURE_SCOPE,
    ]



def combo(net_list,
          kernel_depth,
          kernel_size,
          scope,
          is_training,
          epsilon=1e-5):

  net_list = sync_conv2d(
      net_list, kernel_depth, kernel_size, scope=scope)
  net_list = sync_batch_norm(
      net_list, is_training, name=scope+'/BatchNorm', epsilon=epsilon)
  net_list = sync_relu(net_list)

  return net_list


def separable_combo(inputs_list,
                    filters,
                    is_training,
                    kernel_size=3,
                    rate=1,
                    weight_decay=0.00004,
                    depthwise_weights_initializer_stddev=0.33,
                    pointwise_weights_initializer_stddev=0.06,
                    scope=None,
                    epsilon=1e-5):
  outputs_list = []
  for i, inputs in enumerate(inputs_list):
    with tf.device('/gpu:%d' % i):
      outputs = slim.separable_conv2d(
          inputs, None, kernel_size, depth_multiplier=1,
          weights_initializer=tf.truncated_normal_initializer(
              stddev=depthwise_weights_initializer_stddev),
          rate=rate, weights_regularizer=None,
          scope=scope + '_depthwise', reuse=(i != 0))

    outputs_list.append(outputs)

  outputs_list = sync_batch_norm(
      outputs_list, is_training, name=scope+'_depthwise/BatchNorm',
      epsilon=epsilon)
  outputs_list = sync_relu(outputs_list)

  inputs_list = outputs_list
  outputs_list = []
  for i, inputs in enumerate(inputs_list):
    with tf.device('/gpu:%d' % i):
      outputs = slim.conv2d(inputs,
                            filters,
                            1,
                            weights_initializer=tf.truncated_normal_initializer(
                                stddev=pointwise_weights_initializer_stddev),
                            scope=scope + '_pointwise',
                            reuse=(i != 0))
    outputs_list.append(outputs)

  outputs_list = sync_batch_norm(
      outputs_list, is_training, name=scope+'_pointwise/BatchNorm',
      epsilon=epsilon)
  outputs_list = sync_relu(outputs_list)

  return outputs_list


def aspp(net_list, os16, is_training, fine_tune_batch_norm):
  is_bn_training = (is_training and fine_tune_batch_norm)
  branch_logits = []
  depth = 256
  if os16:
    atrous_rates = [6, 12, 18]
    resize_size = [33, 33]
  else:
    atrous_rates = [12, 24, 36]
    resize_size = [65, 65]

  image_feature_list = sync_avg_pool2d(net_list)
  image_feature_list = combo(
      image_feature_list, depth, 1, 'image_pooling', is_bn_training)
  image_feature_list = sync_resize_bilinear(
      image_feature_list,resize_size)
  for image_feature in image_feature_list:
    branch_logits.append([image_feature])

  aspp_0s = combo(net_list, depth, 1, ASPP_SCOPE + str(0), is_bn_training)
  for pit, aspp in zip(branch_logits, aspp_0s):
    pit.append(aspp)

  for i, rate in enumerate(atrous_rates, 1):
    aspp_features = separable_combo(
        net_list,
        filters=depth,
        is_training=is_bn_training,
        rate=rate,
        scope=ASPP_SCOPE + str(i))

    for pit, aspp in zip(branch_logits, aspp_features):
      pit.append(aspp)

  concat_logits_list = sync_concat(branch_logits)
  concat_logits_list = combo(
      concat_logits_list, depth, 1, CONCAT_PROJECTION_SCOPE, is_bn_training)
  concat_logits_list = sync_dropout(
      concat_logits_list,
      keep_prob=0.9,
      is_training=is_training,
      scope=CONCAT_PROJECTION_SCOPE + '_dropout')

  return concat_logits_list


def decoder(net_list, low_feature_list, is_training):
  with tf.variable_scope(DECODER_SCOPE, DECODER_SCOPE, net_list):
    low_48_list = combo(
        low_feature_list,
        48,
        1,
        'feature_projection' + str(0),
        is_training)

    net_list = sync_resize_bilinear(net_list, [129, 129])

    concat_list = []
    for net, low_48 in zip(net_list, low_48_list):
      concat_list.append([net, low_48])
    decoder_features_list = sync_concat(concat_list)

    decoder_depth = 256
    decoder_features_list = separable_combo(
        decoder_features_list,
        filters=decoder_depth,
        is_training=is_training,
        rate=1,
        scope='decoder_conv0')
    decoder_features_list = separable_combo(
        decoder_features_list,
        filters=decoder_depth,
        is_training=is_training,
        rate=1,
        scope='decoder_conv1')

    return decoder_features_list


def model_fn(samples_list,
             num_of_classes,
             is_training,
             fine_tune_batch_norm,
             os16=True,
             output_affinity=False):
  # Xception
  with slim.arg_scope(
      [slim.conv2d, slim.separable_conv2d],
      weights_initializer=tf.truncated_normal_initializer(stddev=0.09),
      activation_fn=None,
      biases_initializer=None,
      normalizer_fn=None):
    with slim.arg_scope(
        [slim.conv2d],
        weights_regularizer=slim.l2_regularizer(0.00004)):
      net_list, l_list = xception_sync.xception_65(
          samples_list, num_classes=None,
          is_training=(is_training and fine_tune_batch_norm),
          global_pool=False, output_stride=(16 if os16 else 8))
  # ASPP + Decoder
  with slim.arg_scope(
      [slim.conv2d, slim.separable_conv2d],
      weights_regularizer=slim.l2_regularizer(0.00004),
      activation_fn=None,
      normalizer_fn=None,
      biases_initializer=None):
    net_list = aspp(net_list, os16, is_training, fine_tune_batch_norm)
    net_list = decoder(net_list, l_list, (is_training and fine_tune_batch_norm))

  # Logits
  # semantic
  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(0.00004),
      weights_initializer=tf.truncated_normal_initializer(stddev=0.01)):
    with tf.variable_scope(LOGITS_SCOPE_NAME, LOGITS_SCOPE_NAME, net_list):
      semantic_list = sync_conv2d(net_list, num_of_classes, 1, 'semantic')

  semantic_list = sync_resize_bilinear(semantic_list, [513, 513])

  return semantic_list
