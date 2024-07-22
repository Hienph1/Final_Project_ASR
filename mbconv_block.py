import tensorflow as tf
from tensorflow.keras import layers, Model

def mbconv_block(x, filters, kernel_size, strides, expand_ratio, se_ratio, drop_rate):
   inputs = x
   input_filters = x.shape[-1]
   expanded_filters = input_filters * expand_ratio
   
   # Expansion phase
   if expand_ratio != 1:
      x = layers.Conv2D(expanded_filters, kernel_size=1, padding='same', use_bias=False)(x)
      x = layers.BatchNormalization()(x)
      x = layers.Activation('swish')(x)
   
   # Depthwise Convolution
   x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(x)
   x = layers.BatchNormalization()(x)
   x = layers.Activation('swish')(x)
   
   # Squeeze and Excitation
   if se_ratio:
      se_filters = max(1, int(input_filters * se_ratio))
      se = layers.GlobalAveragePooling2D()(x)
      se = layers.Reshape((1, 1, expanded_filters))(se)
      se = layers.Conv2D(se_filters, kernel_size=1, activation='swish', padding='same')(se)
      se = layers.Conv2D(expanded_filters, kernel_size=1, activation='sigmoid', padding='same')(se)
      x = layers.Multiply()([x, se])
   
   # Output phase
   x = layers.Conv2D(filters, kernel_size=1, padding='same', use_bias=False)(x)
   x = layers.BatchNormalization()(x)
   
   # Residual connection
   if strides == 1 and input_filters == filters:
      if drop_rate:
         x = layers.Dropout(drop_rate)(x)
      x = layers.Add()([x, inputs])
   
   return x