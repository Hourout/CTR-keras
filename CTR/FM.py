import tensorflow as tf
K = tf.keras.backend


#FM
class MyLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim=30, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)

    def call(self, x):
        return K.mean(K.pow(K.dot(self.kernel, x), 2)-K.dot(x, K.pow(self.kernel, 2)), 1, keep_dims=True)*0.5

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

def FM(feature_dim):
    inputs = tf.keras.Input((feature_dim,))
    liner = tf.keras.layers.Dense(1)(inputs)
    cross = MyLayer()(inputs)
    add = tf.keras.layers.Add()([liner, cross])
    predictions = tf.keras.Activation('sigmoid')(add)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    return model


#FFM
class MyLayer(tf.keras.layers.Layer):
    def __init__(self, field_dict, field_dim, output_dim=30, **kwargs):
        self.filed_dict = field_dict
        self.filed_dim = field_dim
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.field_dim, self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)

    def call(self, x):
        self.field_cross = K.variable(0, dtype='float32')
        for i in range(input_shape[1]):
            for j in range(i+1, input_shape[1]):
                self.field_cross += K.dot(self.kernel[i, self.field_dict[j]], K.transpose(self.kernel[j, self.field_dict[i]]))*tf.keras.layers.multiply([x[:,i], x[:,j]])
        return self.field_cross

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

def FFM(feature_dim):
    inputs = tf.keras.Input((feature_dim,))
    liner = tf.keras.layers.Dense(1)(inputs)
    cross = MyLayer()(inputs)
    add = tf.keras.layers.Add()([liner, cross])
    predictions = tf.keras.Activation('sigmoid')(add)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    return model