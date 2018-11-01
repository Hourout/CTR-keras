import tensorflow as tf
K = tf.keras.backend


class MyLayer(tf.keras.layers.Layer):
    def __init__(self, field_dict, field_dim, input_dim, output_dim=30, **kwargs):
        self.filed_dict = field_dict
        self.filed_dim = field_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(self.input_dim, self.field_dim, self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)

    def call(self, x):
        self.field_cross = K.variable(0, dtype='float32')
        for i in range(self.input_dim):
            for j in range(i+1, self.input_dim):
                self.field_cross += K.dot(self.kernel[i, self.field_dict[j]],
                    K.transpose(self.kernel[j, self.field_dict[i]]))*tf.keras.layers.multiply([x[:,i], x[:,j]])
        return self.field_cross

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

def FFM(feature_dim, field_dict, field_dim, output_dim=30):
    inputs = tf.keras.Input((feature_dim,))
    liner = tf.keras.layers.Dense(1)(inputs)
    cross = MyLayer(field_dict, field_dim, feature_dim, output_dim)(inputs)
    add = tf.keras.layers.Add()([liner, cross])
    predictions = tf.keras.Activation('sigmoid')(add)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.train.AdamOptimizer(0.001),
                  metrics=['binary_accuracy'])
    return model


