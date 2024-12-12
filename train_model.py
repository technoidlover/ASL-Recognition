import os
from typing import Tuple, Dict
import warnings
import numpy as np
from PIL import Image, ImageFile
from keras.applications import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Configure constants
ImageFile.LOAD_TRUNCATED_IMAGES = True

class GestureClassifier:
    def __init__(self, image_size: int = 224, rgb: bool = False):
        self.image_size = image_size
        self.rgb = rgb
        self.gestures = {
            'L_': 'L', 'fi': 'E', 'ok': 'F', 
            'pe': 'V', 'pa': 'B'
        }
        self.gestures_map = {
            'E': 0, 'L': 1, 'F': 2, 'V': 3, 'B': 4
        }
        self.gesture_names = {v: k for k, v in self.gestures_map.items()}
        
    def process_image(self, path: str) -> np.ndarray:
        """Process single image"""
        with Image.open(path) as img:
            img = img.resize((self.image_size, self.image_size))
            return np.array(img)

    def process_data(self, X_data: np.ndarray, y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize and prepare data for training"""
        X_data = np.array(X_data, dtype='float32')
        if not self.rgb:
            X_data = np.stack((X_data,) * 3, axis=-1)
        X_data /= 255.0
        return X_data, to_categorical(y_data)

    def load_data(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and process all images from directory"""
        X_data, y_data = [], []
        
        for root, _, files in os.walk(image_path):
            for file in files:
                if file.startswith('.'):
                    continue
                    
                path = os.path.join(root, file)
                gesture_name = self.gestures[file[0:2]]
                y_data.append(self.gestures_map[gesture_name])
                X_data.append(self.process_image(path))

        return self.process_data(X_data, np.array(y_data))

    def create_model(self) -> Model:
        """Create and compile the model"""
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(self.image_size, self.image_size, 3)
        )
        
        # Freeze VGG16 layers
        for layer in base_model.layers:
            layer.trainable = False

        # Add custom layers
        x = base_model.output
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        predictions = Dense(len(self.gestures_map), activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, image_path: str, models_path: str, epochs: int = 50):
        """Train the model"""
        # Load and prepare data
        X_data, y_data = self.load_data(image_path)
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, 
            test_size=0.2,
            random_state=42,
            stratify=y_data
        )

        # Create callbacks
        callbacks = [
            ModelCheckpoint(filepath=models_path, save_best_only=True),
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                verbose=1,
                mode='max',
                restore_best_weights=True
            )
        ]

        # Create and train model
        model = self.create_model()
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=64,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history

def main():
    # Initialize and train
    classifier = GestureClassifier(image_size=224, rgb=False)
    model, history = classifier.train(
        image_path='data',
        models_path='models/saved_model.keras'
    )
    
    # Save the trained model
    model.save('models/mymodel.h5')

if __name__ == "__main__":
    main()