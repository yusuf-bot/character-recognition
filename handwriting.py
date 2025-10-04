import tensorflow as tf
import tensorflow_datasets as tfds

# ========================
# 1. Load EMNIST dataset
# ========================
print("Loading EMNIST...")
(ds_train, ds_test), ds_info = tfds.load(
    "emnist/byclass",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

num_classes = ds_info.features["label"].num_classes
print(f"Dataset loaded. Classes: {num_classes}")

# ========================
# 2. Preprocessing
# ========================
def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, -1)  # add channel dim
    return image, label

ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(10000)
ds_train = ds_train.batch(128).prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128).cache().prefetch(tf.data.AUTOTUNE)

# ========================
# 3. Data Augmentation
# ========================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomZoom(0.05),
])

# ========================
# 4. Build Model (VGG-style CNN)
# ========================
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    # Augmentation
    data_augmentation,

    # Block 1
    tf.keras.layers.Conv2D(64, (3,3), padding="same", activation=None),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(64, (3,3), padding="same", activation=None),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.25),

    # Block 2
    tf.keras.layers.Conv2D(128, (3,3), padding="same", activation=None),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(128, (3,3), padding="same", activation=None),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.25),

    # Block 3
    tf.keras.layers.Conv2D(256, (3,3), padding="same", activation=None),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(256, (3,3), padding="same", activation=None),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.25),

    # Dense layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=None),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(num_classes, activation="softmax")
])

# ========================
# 5. Compile Model
# ========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ========================
# 6. Callbacks
# ========================
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5),
    tf.keras.callbacks.ModelCheckpoint("best_emnist.keras", save_best_only=True)
]

# ========================
# 7. Train
# ========================
history = model.fit(
    ds_train,
    validation_data=ds_test,
    epochs=50,
    callbacks=callbacks
)

# ========================
# 8. Save Final Model
# ========================
model.save("final_emnist.keras")
print("✅ Training complete. Model saved as final_emnist.keras")
