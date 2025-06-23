import tensorflow as tf
import numpy as np

x_data = np.array([[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]], dtype=np.float32)
y_data = np.array([[0],
          [0],
          [0],
          [1],
          [1],
          [1]], dtype=np.float32)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_dim=2))
model.add(tf.keras.layers.Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), metrics=['accuracy'])
model.summary()

history = model.fit(x_data, y_data, epochs=5000)
print("Accuracy: ", history.history['accuracy'][-1])