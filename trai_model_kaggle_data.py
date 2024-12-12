import os
import logging
import numpy as np
from PIL import Image
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
IMAGE_SIZE = 224
NUM_CLASSES = 26
BATCH_SIZE = 32
EPOCHS = 50

# Configure paths
MODEL_DIR = 'models'
CHECKPOINT_DIR = os.path.join(MODEL_DIR, 'checkpoints')
MODEL_PATH = os.path.join(MODEL_DIR, 'asl_model.keras')
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'model_{epoch:02d}_{val_accuracy:.2f}.keras')

# Create gesture mappings for lowercase letters
GESTURES = {chr(97+i): i for i in range(26)}  # a-z mapping

def ensure_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def process_image(path):
    """Process single image"""
    try:
        with Image.open(path) as img:
            img = img.convert('RGB')
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            return np.array(img)
    except Exception as e:
        logger.warning(f"Failed to process {path}: {e}")
        return None

def load_dataset(data_path):
    """Load and process entire dataset"""
    if not os.path.exists(data_path):
        raise ValueError(f"Data directory not found: {data_path}")
        
    X_data = []
    y_data = []
    processed = 0
    
    for label in sorted(os.listdir(data_path)):
        label_path = os.path.join(data_path, label)
        if not os.path.isdir(label_path):
            continue
            
        label_idx = GESTURES.get(label.lower())
        if label_idx is None:
            logger.warning(f"Skipping unknown label directory: {label}")
            continue
            
        logger.info(f"Processing directory: {label}")
        for img_file in os.listdir(label_path):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(label_path, img_file)
            img_data = process_image(img_path)
            
            if img_data is not None:
                X_data.append(img_data)
                y_data.append(label_idx)
                processed += 1

    logger.info(f"Processed {processed} images")
    if not X_data:
        raise ValueError("No valid images found")
        
    X_data = np.array(X_data, dtype='float32') / 255.0
    y_data = to_categorical(np.array(y_data), NUM_CLASSES)
    
    return X_data, y_data

def create_model():
    """Create and compile model"""
    base_model = VGG16(weights='imagenet', 
                      include_top=False, 
                      input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    
    # Freeze VGG16 layers
    for layer in base_model.layers:
        layer.trainable = False
        
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def train_model(data_path):
    """Main training pipeline"""
    # Create directories
    ensure_directories()
    
    # Load data
    logger.info("Loading dataset...")
    X_data, y_data = load_dataset(data_path)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42
    )
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy', 
            patience=10, 
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Create and train model
    logger.info("Creating model...")
    model = create_model()
    
    logger.info("Training model...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # Save final model
    model.save(MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")
    
    return model, history

if __name__ == "__main__":
    data_path = "data"  # Update with your dataset path
    model, history = train_model(data_path)