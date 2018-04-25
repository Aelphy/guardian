import keras
import os.path as osp
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, BatchNormalization, Activation, Lambda, Flatten, AveragePooling2D, UpSampling2D
from keras.regularizers import l2
from keras.layers.merge import concatenate

def VGG16_modified(input_shape, num_units_start, num_units_top):
    img_input = Input(input_shape)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten()(x)
    num_units = num_units_start

    while num_units > num_units_top:
        x = Dense(num_units, activation=None, kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(x)
        # x = BatchNormalization()(x) # NOTE: doesn't work with tf optimiser, pnly with Keras one
        x = Activation('elu')(x)

        num_units = num_units // 2

    return keras.models.Model(img_input, x)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def abs_difference(vects):
    x, y = vects
    return K.abs(x - y)

def build(input_shape, num_units_start, num_units_top, feature_extractor):
    '''
        creates the architecture for siamese classification
    '''
    if feature_extractor == 'vgg16':
        model = VGG16_modified(input_shape, num_units_start, num_units_top)
    elif feature_extractor == 'densenet':
        model = DenseNet_modified(input_shape, num_units_start, num_units_top)

    inpt1 = Input(shape=input_shape)
    inpt2 = Input(shape=input_shape)
    output1 = model(inpt1)
    output2 = model(inpt2)
    abs_diff = Lambda(abs_difference)([output1, output2])
    distance = Lambda(euclidean_distance)([output1, output2])

    mlp = Dense(1024, activation='elu', kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(abs_diff)
    mlp = Dense(512, activation='elu', kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(mlp)
    prediction = Dense(1, activation=None, kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(mlp)

    return keras.models.Model([inpt1, inpt2], prediction)
