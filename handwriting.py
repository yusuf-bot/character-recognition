import tensorflow as tf
import tensorflow_datasets as tfds

# ======================
# Load EMNIST/byclass
# ======================
train_ds, test_ds = tfds.load(
    'emnist/byclass',
    split=['train', 'test'],
    as_supervised=True  # gives (image, label) directly
)

num_classes = 62  # EMNIST byclass has digits + upper + lowercase letters

# ======================
# Preprocessing
# ======================
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # normalize to [0,1]
    image = tf.expand_dims(image, -1)           # add channel dim (28,28,1)
    return image, label

batch_size = 128

train_ds = (train_ds
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(10000)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE))

test_ds = (test_ds
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE))

# ======================
# Data Augmentation
# ======================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
])

# ======================
# CNN Model
# ======================
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28,28,1)),

    data_augmentation,  # augmentation only applied on training

    tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
    tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ======================
# Training with Callbacks
# ======================
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
]

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=50,
    callbacks=callbacks
)

# ======================
# Evaluate
# ======================
print("\nFinal Evaluation:")
model.evaluate(test_ds, verbose=2)

# ======================
# Save Model
# ======================
model.save("emnist_cnn.keras")
