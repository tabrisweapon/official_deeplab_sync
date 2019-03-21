# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Xception model.

"Xception: Deep Learning with Depthwise Separable Convolutions"
Fran{\c{c}}ois Chollet
https://arxiv.org/abs/1610.02357

We implement the modified version by Jifeng Dai et al. for their COCO 2017
detection challenge submission, where the model is made deeper and has aligned
features for dense prediction tasks. See their slides for details:

"Deformable Convolutional Networks -- COCO Detection and Segmentation Challenge
2017 Entry"
Haozhi Qi, Zheng Zhang, Bin Xiao, Han Hu, Bowen Cheng, Yichen Wei and Jifeng Dai
ICCV 2017 COCO Challenge workshop
http://presentations.cocodataset.org/COCO17-Detect-MSRA.pdf

We made a few more changes on top of MSRA's modifications:
1. Fully convolutional: All the max-pooling layers are replaced with separable
  conv2d with stride = 2. This allows us to use atrous convolution to extract
  feature maps at any resolution.

2. We support adding ReLU and BatchNorm after depthwise convolution, motivated
  by the design of MobileNetv1.

"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
Applications"
Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang,
Tobias Weyand, Marco Andreetto, Hartwig Adam
https://arxiv.org/abs/1704.04861
"""
import collections
import tensorflow as tf

from deeplab.sync import sync_util

sync_fixed_padding = sync_util.sync_fixed_padding
sync_relu = sync_util.sync_relu
sync_conv2d = sync_util.sync_conv2d
sync_separable_conv2d = sync_util.sync_separable_conv2d
sync_batch_norm = sync_util.sync_batch_norm


slim = tf.contrib.slim

_DEFAULT_MULTI_GRID = [1, 1, 1]


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
  """A named tuple describing an Xception block.

  Its parts are:
    scope: The scope of the block.
    unit_fn: The Xception unit function which takes as input a tensor and
      returns another tensor with the output of the Xception unit.
    args: A list of length equal to the number of units in the block. The list
      contains one dictionary for each unit in the block to serve as argument to
      unit_fn.
  """


def _preprocess_zero_mean_unit_range(inputs_list):
  """Map image values from [0, 255] to [-1, 1]."""
  outputs_list = []
  for i, inputs in enumerate(inputs_list):
    with tf.device('/gpu:%d' % i):
      preprocessed_inputs = (2.0 / 255.0) * tf.to_float(inputs) - 1.0
    outputs_list.append(preprocessed_inputs)
  return outputs_list


@slim.add_arg_scope
def sync_separable_conv2d_same(inputs_list,
                               num_outputs,
                               kernel_size,
                               depth_multiplier,
                               stride,
                               is_training,
                               rate=1,
                               regularize_depthwise=False,
                               scope=None,
                               activated=False):
  """Strided 2-D separable convolution with 'SAME' padding.

  If stride > 1 and use_explicit_padding is True, then we do explicit zero-
  padding, followed by conv2d with 'VALID' padding.

  Note that

     net = separable_conv2d_same(inputs, num_outputs, 3,
       depth_multiplier=1, stride=stride)

  is equivalent to

     net = slim.separable_conv2d(inputs, num_outputs, 3,
       depth_multiplier=1, stride=1, padding='SAME')
     net = resnet_utils.subsample(net, factor=stride)

  whereas

     net = slim.separable_conv2d(inputs, num_outputs, 3, stride=stride,
       depth_multiplier=1, padding='SAME')

  is different when the input's height or width is even, which is why we add the
  current function.

  Consequently, if the input feature map has even height or width, setting
  `use_explicit_padding=False` will result in feature misalignment by one pixel
  along the corresponding dimension.

  Args:
    inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
    num_outputs: An integer, the number of output filters.
    kernel_size: An int with the kernel_size of the filters.
    depth_multiplier: The number of depthwise convolution output channels for
      each input channel. The total number of depthwise convolution output
      channels will be equal to `num_filters_in * depth_multiplier`.
    stride: An integer, the output stride.
    rate: An integer, rate for atrous convolution.
    use_explicit_padding: If True, use explicit padding to make the model fully
      compatible with the open source version, otherwise use the native
      Tensorflow 'SAME' padding.
    regularize_depthwise: Whether or not apply L2-norm regularization on the
      depthwise convolution weights.
    scope: Scope.
    **kwargs: additional keyword arguments to pass to slim.conv2d

  Returns:
    output: A 4-D tensor of size [batch, height_out, width_out, channels] with
      the convolution output.
  """
  padding = 'SAME'
  if stride != 1:
    inputs_list = sync_fixed_padding(inputs_list, kernel_size, rate)
    padding = 'VALID'

  outputs_list = sync_separable_conv2d(inputs_list,
                                       kernel_size,
                                       depth_multiplier=depth_multiplier,
                                       stride=stride,
                                       rate=rate,
                                       padding=padding,
                                       scope=scope + '_depthwise')

  outputs_list = sync_batch_norm(outputs_list, is_training, name=scope+'_depthwise/BatchNorm')

  if activated:
    outputs_list = sync_relu(outputs_list)

  outputs_list = sync_conv2d(outputs_list,
                             num_outputs,
                             1,
                             scope=scope + '_pointwise')

  outputs_list = sync_batch_norm(outputs_list, is_training, name=scope+'_pointwise/BatchNorm')

  if activated:
    outputs_list = sync_relu(outputs_list)

  return outputs_list


@slim.add_arg_scope
def xception_module(inputs_list,
                    depth_list,
                    skip_connection_type,
                    stride,
                    is_training,
                    unit_rate_list=None,
                    rate=1,
                    activation_fn_in_separable_conv=False,
                    regularize_depthwise=False,
                    outputs_collections=None,
                    scope=None,
                    use_bounded_activation=False,
                    use_explicit_padding=True):
  """An Xception module.

  The output of one Xception module is equal to the sum of `residual` and
  `shortcut`, where `residual` is the feature computed by three separable
  convolution. The `shortcut` is the feature computed by 1x1 convolution with
  or without striding. In some cases, the `shortcut` path could be a simple
  identity function or none (i.e, no shortcut).

  Note that we replace the max pooling operations in the Xception module with
  another separable convolution with striding, since atrous rate is not properly
  supported in current TensorFlow max pooling implementation.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth_list: A list of three integers specifying the depth values of one
      Xception module.
    skip_connection_type: Skip connection type for the residual path. Only
      supports 'conv', 'sum', or 'none'.
    stride: The block unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    unit_rate_list: A list of three integers, determining the unit rate for
      each separable convolution in the xception module.
    rate: An integer, rate for atrous convolution.
    activation_fn_in_separable_conv: Includes activation function in the
      separable convolution or not.
    regularize_depthwise: Whether or not apply L2-norm regularization on the
      depthwise convolution weights.
    outputs_collections: Collection to add the Xception unit output.
    scope: Optional variable_scope.
    use_bounded_activation: Whether or not to use bounded activations. Bounded
      activations better lend themselves to quantized inference.
    use_explicit_padding: If True, use explicit padding to make the model fully
      compatible with the open source version, otherwise use the native
      Tensorflow 'SAME' padding.

  Returns:
    The Xception module's output.

  Raises:
    ValueError: If depth_list and unit_rate_list do not contain three elements,
      or if stride != 1 for the third separable convolution operation in the
      residual path, or unsupported skip connection type.
  """
  cheating_list = []
  if len(depth_list) != 3:
    raise ValueError('Expect three elements in depth_list.')
  if unit_rate_list:
    if len(unit_rate_list) != 3:
      raise ValueError('Expect three elements in unit_rate_list.')

  with tf.variable_scope(scope, 'xception_module', inputs_list) as sc:
    residual_list = inputs_list

    # residual branch
    for j in range(3):
      if not activation_fn_in_separable_conv:
        residual_list = sync_relu(residual_list)

      residual_list = sync_separable_conv2d_same(
          residual_list, depth_list[j], 3, depth_multiplier=1,
          stride=stride if j == 2 else 1, is_training=is_training,
          rate=rate*unit_rate_list[j],
          activated=activation_fn_in_separable_conv,
          regularize_depthwise=regularize_depthwise,
          scope='separable_conv' + str(j+1))
      if j==1:
        cheating_list = residual_list

    # merge shortcut branch
    if skip_connection_type == 'conv':
      shortcut_list = sync_conv2d(inputs_list,
                                  depth_list[-1],
                                  [1, 1],
                                  stride=stride,
                                  scope='shortcut')
      shortcut_list = sync_batch_norm(shortcut_list, is_training, name='shortcut/BatchNorm')
      inputs_list = shortcut_list
    if skip_connection_type == 'conv' or skip_connection_type == 'sum':
      outputs_list = []
      i = 0
      for inputs, residual in zip(inputs_list,residual_list):
        with tf.device('/gpu:%d' % i):
          outputs = residual + inputs
        outputs_list.append(outputs)
        i = i + 1
    elif skip_connection_type == 'none':
      outputs_list = residual_list
    else:
      raise ValueError('Unsupported skip connection type.')

    return outputs_list, cheating_list


@slim.add_arg_scope
def stack_blocks_dense(net_list,
                       blocks,
                       output_stride,
                       is_training):
  """Stacks Xception blocks and controls output feature density.

  First, this function creates scopes for the Xception in the form of
  'block_name/unit_1', 'block_name/unit_2', etc.

  Second, this function allows the user to explicitly control the output
  stride, which is the ratio of the input to output spatial resolution. This
  is useful for dense prediction tasks such as semantic segmentation or
  object detection.

  Control of the output feature density is implemented by atrous convolution.

  Args:
    net: A tensor of size [batch, height, width, channels].
    blocks: A list of length equal to the number of Xception blocks. Each
      element is an Xception Block object describing the units in the block.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution, which needs to be equal to
      the product of unit strides from the start up to some level of Xception.
      For example, if the Xception employs units with strides 1, 2, 1, 3, 4, 1,
      then valid values for the output_stride are 1, 2, 6, 24 or None (which
      is equivalent to output_stride=24).
    outputs_collections: Collection to add the Xception block outputs.

  Returns:
    net: Output tensor with stride equal to the specified output_stride.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  # The current_stride variable keeps track of the effective stride of the
  # activations. This allows us to invoke atrous convolution whenever applying
  # the next residual unit would result in the activations having stride larger
  # than the target output_stride.
  current_stride = 1

  # The atrous convolution rate parameter.
  rate = 1
  c_list=[]
  l_list=[]
  for b_index, block in enumerate(blocks):
    with tf.variable_scope(block.scope, 'block', net_list) as sc:
      for i, unit in enumerate(block.args):
        if output_stride is not None and current_stride > output_stride:
          raise ValueError('The target output_stride cannot be reached.')
        with tf.variable_scope('unit_%d' % (i + 1), values=net_list):
          # If we have reached the target output_stride, then we need to employ
          # atrous convolution with stride=1 and multiply the atrous rate by the
          # current unit's stride for use in subsequent layers.
          if output_stride is not None and current_stride == output_stride:
            net_list, c_list = block.unit_fn(
                net_list, rate=rate,
                is_training=is_training, **dict(unit, stride=1))
            rate *= unit.get('stride', 1)
          else:
            net_list, c_list = block.unit_fn(
                net_list, rate=1, is_training=is_training, **unit)
            current_stride *= unit.get('stride', 1)
    if b_index==1:
      l_list = c_list
  if output_stride is not None and current_stride != output_stride:
    raise ValueError('The target output_stride cannot be reached.')

  return net_list, l_list


def root_block(inputs_list, is_training):
  # conv 1_1
  outputs_list = sync_conv2d(inputs_list, 32, 3, stride=2, padding='SAME',
                             scope='entry_flow/conv1_1')
  outputs_list = sync_batch_norm(outputs_list, is_training, name='entry_flow/conv1_1/BatchNorm')
  outputs_list = sync_relu(outputs_list)

  # conv 1_2
  outputs_list = sync_fixed_padding(outputs_list, 3)
  outputs_list = sync_conv2d(outputs_list, 64, 3, padding='VALID',
                             scope='entry_flow/conv1_2')
  outputs_list = sync_batch_norm(outputs_list, is_training, name='entry_flow/conv1_2/BatchNorm')
  outputs_list = sync_relu(outputs_list)
  return outputs_list



def xception(inputs_list,
             blocks,
             num_classes=None,
             is_training=True,
             global_pool=True,
             keep_prob=0.5,
             output_stride=None,
             reuse=None,
             scope=None):
  """Generator for Xception models.

  This function generates a family of Xception models. See the xception_*()
  methods for specific model instantiations, obtained by selecting different
  block instantiations that produce Xception of various depths.

  Args:
    inputs_list: A list of tensors of size [batch, height_in, width_in, channels]. Must be
      floating point. If a pretrained checkpoint is used, pixel values should be
      the same as during training (see go/slim-classification-models for
      specifics).
    blocks: A list of length equal to the number of Xception blocks. Each
      element is an Xception Block object describing the units in the block.
    num_classes: Number of predicted classes for classification tasks.
      If 0 or None, we return the features before the logit layer.
    is_training: whether batch_norm layers are in training mode.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    keep_prob: Keep probability used in the pre-logits dropout layer.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is 0 or None,
      then net is the output of the last Xception block, potentially after
      global average pooling. If num_classes is a non-zero integer, net contains
      the pre-softmax activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  inputs_list = _preprocess_zero_mean_unit_range(inputs_list)
  with tf.variable_scope(scope, 'xception', inputs_list) as sc:

      net_list = inputs_list
      if output_stride is not None:
        if output_stride % 2 != 0:
          raise ValueError('The output_stride needs to be a multiple of 2.')
        output_stride /= 2
      # Root block function operated on inputs.
      net_list = root_block(net_list, is_training)
      # Extract features for entry_flow, middle_flow, and exit_flow.
      net_list, low_feature_list = stack_blocks_dense(
          net_list, blocks, output_stride, is_training=is_training)

      return net_list, low_feature_list


def xception_block(scope,
                   depth_list,
                   skip_connection_type,
                   activation_fn_in_separable_conv,
                   regularize_depthwise,
                   num_units,
                   stride,
                   unit_rate_list=None):
  """Helper function for creating a Xception block.

  Args:
    scope: The scope of the block.
    depth_list: The depth of the bottleneck layer for each unit.
    skip_connection_type: Skip connection type for the residual path. Only
      supports 'conv', 'sum', or 'none'.
    activation_fn_in_separable_conv: Includes activation function in the
      separable convolution or not.
    regularize_depthwise: Whether or not apply L2-norm regularization on the
      depthwise convolution weights.
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the last unit.
      All other units have stride=1.
    unit_rate_list: A list of three integers, determining the unit rate in the
      corresponding xception block.

  Returns:
    An Xception block.
  """
  if unit_rate_list is None:
    unit_rate_list = _DEFAULT_MULTI_GRID
  return Block(scope, xception_module, [{
      'depth_list': depth_list,
      'skip_connection_type': skip_connection_type,
      'activation_fn_in_separable_conv': activation_fn_in_separable_conv,
      'regularize_depthwise': regularize_depthwise,
      'stride': stride,
      'unit_rate_list': unit_rate_list,
  }] * num_units)


def xception_65(inputs_list,
                num_classes=None,
                is_training=True,
                global_pool=True,
                keep_prob=0.5,
                output_stride=None,
                regularize_depthwise=False,
                multi_grid=None,
                reuse=None,
                scope='xception_65'):
  """Xception-65 model."""
  blocks = [
      xception_block('entry_flow/block1',
                     depth_list=[128, 128, 128],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2),
      xception_block('entry_flow/block2',
                     depth_list=[256, 256, 256],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2),
      xception_block('entry_flow/block3',
                     depth_list=[728, 728, 728],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2),
      xception_block('middle_flow/block1',
                     depth_list=[728, 728, 728],
                     skip_connection_type='sum',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=16,
                     stride=1),
      xception_block('exit_flow/block1',
                     depth_list=[728, 1024, 1024],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2),
      xception_block('exit_flow/block2',
                     depth_list=[1536, 1536, 2048],
                     skip_connection_type='none',
                     activation_fn_in_separable_conv=True,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=1,
                     unit_rate_list=multi_grid),
  ]
  return xception(inputs_list,
                  blocks=blocks,
                  num_classes=num_classes,
                  is_training=is_training,
                  global_pool=global_pool,
                  keep_prob=keep_prob,
                  output_stride=output_stride,
                  reuse=reuse,
                  scope=scope)
