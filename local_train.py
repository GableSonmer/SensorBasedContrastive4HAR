# model = tf.keras.models.Sequential([
#     Conv1D(128, 5, activation='relu', input_shape=(data.shape[1], data.shape[2])),
#     MaxPooling1D(2),
#     Conv1D(128, 5, activation='relu'),
#     MaxPooling1D(2),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(label.shape[1])
# ])
# loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
# model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
#
# model.fit(data, label, epochs=100, batch_size=32, validation_split=0.2)