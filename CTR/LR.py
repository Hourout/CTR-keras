import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def lr_model():
    inputs = tf.keras.Input((30,))
    pred = tf.keras.layers.Dense(units=1, 
                                 bias_regularizer=tf.keras.regularizers.l2(0.01),
                                 kernel_regularizer=tf.keras.regularizers.l1(0.02),
                                 activation=tf.nn.sigmoid)(inputs)
    lr = tf.keras.Model(inputs, pred)
    lr.compile(loss='binary_crossentropy',
               optimizer=tf.train.AdamOptimizer(0.001),
               metrics=['binary_accuracy'])
    return lr

def train():
    lr = lr_model()
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2,
                                                        random_state=27, stratify=data.target)
    lr.fit(X_train, y_train, epochs=3, batch_size=16, validation_data=(X_test, y_test))
    return lr

if __name__ == '__main__':
    lr = train()
