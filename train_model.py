import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Einstellungen

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "archive" / "wastes"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 25

MODEL_PATH = BASE_DIR / "muell_model_projekt.keras"

CLASS_NAMES = ["glas", "metall", "papier", "plastik", "restmuell"]

tf.random.set_seed(42)
np.random.seed(42)

# Explorative Datenanalyse: Anzahl der Bilder pro Klasse

def count_images(folder):
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    counts = {}

    for class_name in CLASS_NAMES:
        class_folder = folder / class_name
        count = 0

        for file in class_folder.iterdir():
            if file.suffix.lower() in image_extensions:
                count += 1

        counts[class_name] = count

    return counts


train_counts = count_images(TRAIN_DIR)
test_counts = count_images(TEST_DIR)

print("\nEDA: Anzahl der Bilder pro Klasse")
print("--------------------------------")

for class_name in CLASS_NAMES:
    print(
        f"{class_name}: "
        f"Training = {train_counts[class_name]}, "
        f"Test = {test_counts[class_name]}"
    )

plt.figure()
plt.bar(train_counts.keys(), train_counts.values())
plt.title("EDA: Anzahl der Trainingsbilder pro Klasse")
plt.xlabel("Klasse")
plt.ylabel("Anzahl Bilder")
plt.show()

plt.figure()
plt.bar(test_counts.keys(), test_counts.values())
plt.title("EDA: Anzahl der Testbilder pro Klasse")
plt.xlabel("Klasse")
plt.ylabel("Anzahl Bilder")
plt.show()

# Daten laden

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# Datenaugmentation

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Transfer Learning mit MobileNetV2

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# Modell erstellen

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),

    data_augmentation,
    layers.Rescaling(1. / 127.5, offset=-1),

    base_model,

    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(len(CLASS_NAMES), activation="softmax")
])

# Modell kompilieren

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModellübersicht:")
model.summary()

# Restmüll stärker gewichten

class_weight = {
    0: 1.0,   # glas
    1: 1.0,   # metall
    2: 1.0,   # papier
    3: 1.0,   # plastik
    4: 2.5    # restmuell wichtiger
}

# Modell trainieren

print("\nTraining startet...")

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    class_weight=class_weight
)

# Diagramme anzeigen

plt.figure()
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Test")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Test")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Confusion Matrix erstellen

y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

disp.plot()
plt.title("Confusion Matrix")
plt.show()

# Modell speichern

model.save(MODEL_PATH)

print(f"\nModell wurde gespeichert unter: {MODEL_PATH}")

# Einzelbild klassifizieren

def classify_image(image_path):

    img = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(prediction)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = np.max(prediction) * 100

    print("--------------------------------")
    print(f"Bilddatei: {image_path.name}")
    print(f"Klassifikation: {predicted_class}")
    print(f"Sicherheit: {confidence:.2f} %")

# Testbilder analysieren
print("\nAnalyse der einzelnen Testbilder:")

for i in range(1, 6):
    image_path = BASE_DIR / f"test{i}.jpg"

    if image_path.exists():
        classify_image(image_path)
    else:
        print("--------------------------------")
        print(f"test{i}.jpg wurde nicht gefunden.")

