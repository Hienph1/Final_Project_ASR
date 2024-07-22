from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2

def YOHO(nb_filters, nb_strides):
   inputs = layers.Input(shape=(256,40))
   x = inputs
   x = layers.Reshape((256,40,1))(x)
   x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', use_bias=False,
                     activation=None, kernel_regularizer=l2(1e-3), bias_regularizer=l2(1e-3))(x)
   x = layers.BatchNormalization(center=True, scale=False, epsilon=1e-4)(x)
   x = layers.Activation('relu')(x)

   for nb_filter, nb_stride in zip(nb_filters, nb_strides):
      x = layers.DepthwiseConv2D(kernel_size=3, strides=nb_stride, depth_multiplier=1, padding='same', use_bias=False,
                                 activation=None)(x)
      x = layers.BatchNormalization(center=True, scale=False, epsilon=1e-4)(x)
      x = layers.ReLU()(x)
      x = layers.Conv2D(filters=nb_filter, kernel_size=1, strides=1, padding='same', use_bias=False,
                        activation=None, kernel_regularizer=l2(1e-2), bias_regularizer=l2(1e-2))(x)
      x = layers.BatchNormalization(center=True, scale=False, epsilon=1e-4)(x)
      x = layers.ReLU()(x)

      x = layers.SpatialDropout2D(0.3)(x)

   model = models.Model(inputs=inputs, outputs=x)
   return model