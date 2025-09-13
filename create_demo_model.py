import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

# Create a tiny demo CNN model
model = Sequential([
    Conv2D(8, (3,3), activation='relu', input_shape=(299,299,3)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(1, activation='sigmoid')  # output: 0=real, 1=fake
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Save demo model
model.save('deepfake_model_demo.h5')
print("Demo model saved as deepfake_model_demo.h5")
