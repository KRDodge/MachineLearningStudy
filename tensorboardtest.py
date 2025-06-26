import datetime
import numpy as np
import os
import tensorflow as tf
import tensorboard

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=2, input_dim=2))
tf.model.add(tf.keras.layers.Activation('sigmoid'))
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=2))
tf.model.add(tf.keras.layers.Activation('sigmoid'))
tf.model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.SGD(learning_rate=0.1),  metrics=['accuracy'])
tf.model.summary()

log_dir = os.path.join(".", "logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = tf.model.fit(x_data, y_data, epochs=10000, callbacks=[tensorboard_callback])

predictions = tf.model.predict(x_data)
print('Prediction: \n', predictions)

score = tf.model.evaluate(x_data, y_data)
print('Accuracy: ', score[1])