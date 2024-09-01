import sys
import tensorflow as tf
import tensorflow_datasets as tfds

# Use MNIST handwriting dataset
train_ds, test_ds = tfds.load('BinaryAlphaDigits', split=['train[:70%]', 'train[70%:]'], shuffle_files=True)

x_train = train_ds.map(lambda i: i['image'])
y_train = train_ds.map(lambda l: l['label'])
x_test = test_ds.map(lambda x: x['image'])
y_test = test_ds.map(lambda y: y['label'])


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=(20, 16, 1)
    ),

    # Max-pooling layer, using 2x2 pool size
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    # Add an output layer with output units for all 10 digits
    tf.keras.layers.Dense(36, activation="softmax")])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(tf.data.Dataset.zip((x_train, y_train)).batch(1), epochs=200)

# Evaluate neural network performance
model.evaluate(tf.data.Dataset.zip((x_train, y_train)).batch(1), verbose=2)




model.save('brad.keras')