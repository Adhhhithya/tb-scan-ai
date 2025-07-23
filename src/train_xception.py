import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# Load preprocessed data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
X = np.load(os.path.join(BASE_DIR, "../model/X_xception.npy"))
y = np.load(os.path.join(BASE_DIR, "../model/y_xception.npy"))

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Xception expects input shape (299, 299, 3)
input_tensor = Input(shape=(299, 299, 3))

# Load base Xception model (exclude top)
base_model = Xception(weights='imagenet', include_top=False, input_tensor=input_tensor)
base_model.trainable = False  # Freeze base layers

# Add custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
checkpoint_path = os.path.join(BASE_DIR, "../model/best_xception_model.h5")
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=16,
    callbacks=[early_stop, checkpoint]
)

# Save final model (optional)
model.save(os.path.join(BASE_DIR, "../model/xception_tb_model.h5"))
print("✅ Xception model training complete and saved!")
