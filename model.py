import tensorflow as tf
import numpy as np


a = np.random.random((20000,1))*100-50
aa = a**2

x_train = np.array(a)
y_train = np.array(aa)

#defining architecture   Can change Sequential on Sigmoid or Relu
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(20, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001), input_shape=(1, )),
    tf.keras.layers.Dense(20, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dense(1)
])
#reshaping x_train so its 2dimensional
#x_train = x_train.reshape(-1, 1)
#compiling model
#sgd - GradientDescent (градиентный спуск)
#mse - MeanSquadError (среднеквадратическая ошибка)
mse = tf.keras.losses.MeanSquaredError()
model.compile(optimizer=tf.keras.optimizers.AdamW(), loss=tf.keras.losses.MeanSquaredError())

#training parameters
epochs = 5000
batch_size = 256

#Model training
model.fit(x_train, y_train, validation_split=0.2,epochs=epochs, batch_size=batch_size)

#Accuracy check (loss check)
loss = model.evaluate(x_train, y_train)
#print(f"Loss: {loss}")

#Prediction (Prognose)
#predictions = model.predict(x_train)
#print(f"Predictions: {predictions}")
print(model.predict([2, -2, 7, 15, 20, -5]))
