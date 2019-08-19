import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
K = tf.keras.backend


class MyLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim=30, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(self.input_dim, self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)

    def call(self, x):
        a = K.pow(K.dot(x,self.kernel), 2)
        b = K.dot(K.pow(x, 2), K.pow(self.kernel, 2))
        return K.mean(a-b, 1, keepdims=True)*0.5

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

def FM(feature_dim):
    inputs = tf.keras.Input((feature_dim,))
    liner = tf.keras.layers.Dense(units=1, 
                                  bias_regularizer=tf.keras.regularizers.l2(0.01),
                                  kernel_regularizer=tf.keras.regularizers.l1(0.02),
                                  )(inputs)
    cross = MyLayer(feature_dim)(inputs)
    add = tf.keras.layers.Add()([liner, cross])
    predictions = tf.keras.layers.Activation('sigmoid')(add)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.train.AdamOptimizer(0.001),
                  metrics=['binary_accuracy'])
    return model

def train():
    fm = FM(30)
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2,
                                                        random_state=27, stratify=data.target)
    fm.fit(X_train, y_train, epochs=3, batch_size=16, validation_data=(X_test, y_test))
    return fm


if __name__ == '__main__':
    fm = train()
