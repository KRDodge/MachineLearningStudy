import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

# Seed 설정
tf.random.set_seed(777)
np.random.seed(777)
random.seed(777)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 모델 정의
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(784,), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=15, batch_size=100, verbose=2)

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc:.4f}")

r = random.randint(0, x_test.shape[0] - 1)
pred = model.predict(np.expand_dims(x_test[r], axis=0), verbose=0)

print("Label: ", np.argmax(y_test[r]))
print("Prediction: ", np.argmax(pred))

plt.imshow(x_test[r].reshape(28, 28), cmap="Greys")
plt.show()