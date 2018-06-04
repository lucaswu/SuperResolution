from layers import *
from configs import *


def create_model(name, patches):
    if name == 'srcnn':
        return srcnn_935(patches)
    elif name == 'vgg7':
        return vgg7(patches)
    else:
        return vgg_deconv_7(patches)


def srcnn_935(patches, name='srcnn'):
    with tf.variable_scope(name):
        upscaled_patches = tf.image.resize_bicubic(patches, [INPUT_SIZE, INPUT_SIZE], True)
        conv1 = conv2d(upscaled_patches, 9, 9, 64, padding='VALID', name='conv1')
        relu1 = relu(conv1, name='relu1')
        conv2 = conv2d(relu1, 3, 3, 32, padding='VALID', name='conv2')
        relu2 = relu(conv2, name='relu2')
        return conv2d(relu2, 5, 5, NUM_CHENNELS, padding='VALID', name='conv3')


def vgg7(patches, name='vgg7'):
    """
    模型的输出
    :param patches: input patches to improve resolution. must has format of
        [batch_size, patch_height, patch_width, patch_chennels]
    :param name: the name of the network
    :return: the RSCNN inference function
    """
    with tf.variable_scope(name):
        upscaled_patches = tf.image.resize_bicubic(patches, [INPUT_SIZE, INPUT_SIZE], True)
        conv1 = conv2d(upscaled_patches, 3, 3, 32, padding='VALID', name='conv1')
        lrelu1 = leaky_relu(conv1, name='leaky_relu1')
        conv2 = conv2d(lrelu1, 3, 3, 32, padding='VALID', name='conv2')
        lrelu2 = leaky_relu(conv2, name='leaky_relu2')
        conv3 = conv2d(lrelu2, 3, 3, 64, padding='VALID', name='conv3')
        lrelu3 = leaky_relu(conv3, name='leaky_relu3')
        conv4 = conv2d(lrelu3, 3, 3, 64, padding='VALID', name='conv4')
        lrelu4 = leaky_relu(conv4, name='leaky_relu4')
        conv5 = conv2d(lrelu4, 3, 3, 128, padding='VALID', name='conv5')
        lrelu5 = leaky_relu(conv5, name='leaky_relu5')
        conv6 = conv2d(lrelu5, 3, 3, 128, padding='VALID', name='conv6')
        lrelu6 = leaky_relu(conv6, name='leaky_relu6')
        return conv2d(lrelu6, 3, 3, NUM_CHENNELS, padding='VALID', name='conv_out')


def vgg_deconv_7(patches, name='vgg_deconv_7'):
    with tf.variable_scope(name):
        conv1 = conv2d(patches, 3, 3, 16, padding='VALID', name='conv1')
        lrelu1 = leaky_relu(conv1, name='leaky_relu1')
        conv2 = conv2d(lrelu1, 3, 3, 32, padding='VALID', name='conv2')
        lrelu2 = leaky_relu(conv2, name='leaky_relu2')
        conv3 = conv2d(lrelu2, 3, 3, 64, padding='VALID', name='conv3')
        lrelu3 = leaky_relu(conv3, name='leaky_relu3')
        conv4 = conv2d(lrelu3, 3, 3, 128, padding='VALID', name='conv4')
        lrelu4 = leaky_relu(conv4, name='leaky_relu4')
        conv5 = conv2d(lrelu4, 3, 3, 128, padding='VALID', name='conv5')
        lrelu5 = leaky_relu(conv5, name='leaky_relu5')
        conv6 = conv2d(lrelu5, 3, 3, 256, padding='VALID', name='conv6')
        lrelu6 = leaky_relu(conv6, name='leaky_relu6')

        batch_size = int(lrelu6.get_shape()[0])
        rows = int(lrelu6.get_shape()[1])
        cols = int(lrelu6.get_shape()[2])
        channels = int(patches.get_shape()[3])
        # to avoid chessboard artifacts, the filter size must be dividable by the stride
        return deconv2d(lrelu6, 4, 4, [batch_size, rows*2, cols*2, channels], stride=(2, 2), name='deconv_out')


def loss(inferences, ground_truthes, huber_width=0.1, weights_decay=0, name='loss'):
    with tf.name_scope(name):
        slice_begin = (int(ground_truthes.get_shape()[1]) - int(inferences.get_shape()[1])) // 2
        slice_end = int(inferences.get_shape()[1]) + slice_begin
        delta = inferences - ground_truthes[:, slice_begin: slice_end, slice_begin: slice_end, :]

        delta *= [[[[0.11448, 0.58661, 0.29891]]]]  # weights of B, G and R
        l2_loss = tf.pow(delta, 2)
        mse_loss = tf.reduce_mean(tf.reduce_sum(l2_loss, axis=[1, 2, 3]))

        if weights_decay > 0:
            weights = tf.get_collection('weights')
            reg_loss = weights_decay * tf.reduce_sum(
                tf.pack([tf.nn.l2_loss(i) for i in weights]), name='regularization_loss')
            return mse_loss + reg_loss
        else:
            return mse_loss
