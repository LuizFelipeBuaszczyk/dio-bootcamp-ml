from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import seaborn as sns
import pandas as pd

(train_image, train_labels), (test_image, test_labels) = datasets.mnist.load_data()

train_image = train_image.reshape((60000, 28, 28, 1))
test_image = test_image.reshape((10000, 28, 28, 1))

train_image, test_image = train_image / 255.0, test_image / 255.0

classes =[0,1,2,3,4,5,6,7,8,9]

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

y_true = test_labels
y_pred_probs = model.predict(test_image)  # Probabilidades
y_pred = np.argmax(y_pred_probs, axis=1)  # Classes previstas


con_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

## Lendo a matriz e realizando o cálculo das métricas
num_classes = len(classes)

N = np.sum(con_mat)
for i in range(num_classes):
    VP = con_mat[i, i]
    FP = np.sum(con_mat[:, i]) - VP
    FN = np.sum(con_mat[i, :]) - VP
    VN = np.sum(con_mat) - VP - FP - FN


sens = VP / (VP+FN)
espec = VN / (FP+VN)
accuracy = (VP+VN) / N
precision = VP / (VP+FP)
f_score = 2 * (precision * sens) / (precision + sens)

print(f'\Sensibilidade: {sens:.4f}')
print(f'Especificidade: {espec:.4f}')
print(f'Acurácia: {accuracy:.4f}')
print(f"Precissão: {precision:.4f}")
print(f'F-Score: {f_score:.4f}')
