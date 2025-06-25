import tensorflow as tf
import numpy as np

x_data = np.array([[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]], dtype=np.float32)
y_data = np.array([[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]], dtype=np.float32)

x_test = np.array([[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]], dtype=np.float32)
y_test = np.array([[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]], dtype=np.float32)

learning_rate = 0.1

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=3, input_dim=3, activation='softmax'))
tf.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), metrics=['accuracy'])

tf.model.fit(x_data, y_data, epochs=1000)

# predict
print("Prediction: ", tf.model.predict_step(x_test))

# Calculate the accuracy
print("Accuracy: ", tf.model.evaluate(x_test, y_test)[1])