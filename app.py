from flask import Flask, request, jsonify, render_template, Response
import tensorflow as tf
import numpy as np
import os
import io
import json
import shutil
from PIL import Image, ImageDraw

# Configure TensorFlow to be memory-efficient
print("Configuring TensorFlow memory settings...")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Memory growth setting error: {e}")

# Limit CPU memory usage
try:
    tf.config.set_logical_device_configuration(
        tf.config.list_physical_devices('CPU')[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=384)]
    )
except:
    print("Could not set memory limit for TensorFlow")

app = Flask(__name__)

# CORS headers
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Max-Age', '3600')
    return response

# Handle OPTIONS requests for CORS preflight
@app.route('/', methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def options_handler(path=None):
    return jsonify({}), 200

# Global variables
model = None
IMAGE_SIZE = 128  # Reduced from 224 for memory efficiency
diseases = [
    'Normal',
    'Catarata',
    'Glaucoma',
    'Retinopatía diabética',
    'Degeneración macular',
    'Conjuntivitis'
]

# Base de conocimiento para recomendaciones
def create_disease_info():
    return {
        'Normal': {
            'description': 'No se detecta enfermedad ocular.',
            'medications': ['N/A'],
            'treatment': 'Revisiones periódicas con oftalmólogo.'
        },
        'Catarata': {
            'description': 'Opacidad del cristalino que causa visión borrosa.',
            'medications': ['Gotas con antioxidantes', 'Vitaminas A, C y E'],
            'treatment': 'Cirugía de cataratas para reemplazar el cristalino.'
        },
        'Glaucoma': {
            'description': 'Daño al nervio óptico generalmente por presión ocular elevada.',
            'medications': ['Timolol', 'Latanoprost', 'Brimonidina'],
            'treatment': 'Medicamentos para reducir la presión ocular, láser o cirugía.'
        },
        'Retinopatía diabética': {
            'description': 'Daño a los vasos sanguíneos de la retina causado por diabetes.',
            'medications': ['Control de glucosa', 'Inyecciones anti-VEGF'],
            'treatment': 'Control de diabetes, fotocoagulación con láser, vitrectomía.'
        },
        'Degeneración macular': {
            'description': 'Deterioro de la mácula que causa pérdida de visión central.',
            'medications': ['Suplementos AREDS2', 'Inyecciones anti-VEGF'],
            'treatment': 'Suplementos nutricionales, inyecciones intravítreas, terapia fotodinámica.'
        },
        'Conjuntivitis': {
            'description': 'Inflamación de la conjuntiva, puede ser viral, bacteriana o alérgica.',
            'medications': ['Antibióticos tópicos', 'Antihistamínicos', 'Lágrimas artificiales'],
            'treatment': 'Medicamentos según la causa, compresas frías, higiene ocular.'
        }
    }

# Create a more efficient model with fewer parameters
def create_efficient_model():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    
    # Create a smaller, more efficient model
    model = Sequential([
        # First convolutional block - fewer filters
        Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        MaxPooling2D(2, 2),
        
        # Second convolutional block
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        
        # Flatten and dense layers - smaller dense layer
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.4),  # Slightly less dropout
        Dense(len(diseases), activation='softmax')
    ])
    
    # Use a lower learning rate for stability
    optimizer = Adam(learning_rate=0.0005)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Model created with {model.count_params()} parameters")
    return model

# Function to load the model
def load_model():
    global model
    import gc
    
    # Clear any existing model from memory
    if model is not None:
        del model
        gc.collect()
        tf.keras.backend.clear_session()
    
    print("Starting model loading process...")
    try:
        # Try to load the model if it exists
        model_paths = [
            'eye_disease_model.h5',
            '/tmp/eye_disease_model.h5'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                print(f"Loading model from {path}...")
                model = tf.keras.models.load_model(path)
                print(f"Model loaded successfully from {path}")
                return
        
        # If no model exists, create and train a new one
        print("No existing model found, creating a new one...")
        model = create_efficient_model()
        
        # Create and train with minimal test data
        create_minimal_test_dataset()
        train_model_with_minimal_data()
        
        # Save the model
        try:
            model.save('/tmp/eye_disease_model.h5')
            print("Model saved to /tmp/eye_disease_model.h5")
        except Exception as save_error:
            print(f"Error saving model: {save_error}")
    
    except Exception as e:
        print(f"Error loading/creating model: {e}")
        import traceback
        traceback.print_exc()
        
        # Last resort - create a minimal model without training
        try:
            print("Creating emergency fallback model...")
            model = create_efficient_model()
        except Exception as fallback_error:
            print(f"Critical error creating fallback model: {fallback_error}")
            model = None

# Create minimal test dataset with fewer images
def create_minimal_test_dataset():
    base_dir = '/tmp/dataset'
    
    # Create directories
    os.makedirs(f'{base_dir}/train', exist_ok=True)
    os.makedirs(f'{base_dir}/validation', exist_ok=True)
    
    # Create test data for each disease - only 3 images per class
    for disease in diseases:
        os.makedirs(f'{base_dir}/train/{disease}', exist_ok=True)
        os.makedirs(f'{base_dir}/validation/{disease}', exist_ok=True)
        
        # Create color based on disease
        if disease == 'Normal':
            color = (200, 200, 200)  # Gris claro
        elif disease == 'Catarata':
            color = (200, 200, 150)  # Amarillento
        elif disease == 'Glaucoma':
            color = (150, 150, 200)  # Azulado
        elif disease == 'Retinopatía diabética':
            color = (200, 150, 150)  # Rojizo
        elif disease == 'Degeneración macular':
            color = (150, 200, 150)  # Verdoso
        else:  # Conjuntivitis
            color = (200, 150, 200)  # Rosado
        
        # Create only 3 images per disease to save memory
        for i in range(3):
            # Create image
            img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color=color)
            draw = ImageDraw.Draw(img)
            
            # Add eye simulation - scaled for smaller image
            center = IMAGE_SIZE // 2
            radius = IMAGE_SIZE // 4
            draw.ellipse((center-radius, center-radius, center+radius, center+radius), fill=(255, 255, 255))
            draw.ellipse((center-radius//2, center-radius//2, center+radius//2, center+radius//2), fill=(0, 0, 0))
            
            # Save image
            img.save(f'{base_dir}/train/{disease}/{i+1}.jpg')
    
    # Move 1 image to validation for each disease
    for disease in diseases:
        files = os.listdir(f'{base_dir}/train/{disease}')
        if files:
            shutil.move(
                f'{base_dir}/train/{disease}/{files[0]}',
                f'{base_dir}/validation/{disease}/{files[0]}'
            )

# Train model with minimal data
def train_model_with_minimal_data():
    if model is None:
        print("Cannot train: model is None")
        return
    
    base_dir = '/tmp/dataset'
    
    # Create data generators with minimal augmentation
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    # Use small batch size
    batch_size = 4
    
    train_generator = train_datagen.flow_from_directory(
        f'{base_dir}/train',
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        f'{base_dir}/validation',
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    # Train for minimal epochs
    print("Training model with minimal data...")
    model.fit(
        train_generator,
        steps_per_epoch=max(1, train_generator.samples // batch_size),
        validation_data=validation_generator,
        validation_steps=max(1, validation_generator.samples // batch_size),
        epochs=2,  # Minimal epochs
        verbose=1
    )
    
    print("Minimal training completed")

# Root endpoint
@app.route('/')
def index():
    return jsonify({
        'status': 'success',
        'message': 'API de detección de enfermedades oculares funcionando correctamente'
    })

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    print("Received prediction request")
    
    # Check if model is loaded
    global model
    if model is None:
        print("Model is None, attempting to load it")
        load_model()
        
        # If still None after loading attempt, return error
        if model is None:
            print("Failed to load model, returning error")
            return jsonify({'error': 'El modelo no está disponible en este momento. Por favor, inténtelo más tarde.'}), 503
    
    if 'image' not in request.files:
        print("No image found in request")
        return jsonify({'error': 'No se encontró imagen en la solicitud'}), 400
    
    file = request.files['image']
    if file.filename == '':
        print("Empty filename")
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
    
    try:
        # Log request details
        print(f"Processing image: {file.filename}, Content-Type: {file.content_type}")
        
        # Read file content once
        file_content = file.read()
        print(f"Image size: {len(file_content)} bytes")
        
        # Process the image
        img = Image.open(io.BytesIO(file_content))
        print(f"Image format: {img.format}, mode: {img.mode}, size: {img.size}")
        
        img = img.convert('RGB')  # Ensure RGB format
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Make prediction
        print("Making prediction...")
        prediction = model.predict(img_array, batch_size=1)
        disease_index = np.argmax(prediction[0])
        confidence = prediction[0][disease_index] * 100
        
        disease_name = diseases[disease_index]
        disease_info = create_disease_info()[disease_name]
        
        print(f"Prediction result: {disease_name} with {confidence:.2f}% confidence")
        
        # Clean up to save memory
        import gc
        del img_array
        gc.collect()
        
        result = {
            'disease': disease_name,
            'confidence': f"{confidence:.2f}%",
            'description': disease_info['description'],
            'medications': disease_info['medications'],
            'treatment': disease_info['treatment']
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in predict endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error al procesar la imagen: {str(e)}'}), 500

# Training endpoint
@app.route('/train', methods=['POST'])
def train():
    if 'image' not in request.files or 'label' not in request.form:
        return jsonify({'error': 'Se requiere imagen y etiqueta'}), 400
    
    file = request.files['image']
    label = request.form['label']
    
    if file.filename == '' or label == '':
        return jsonify({'error': 'Archivo o etiqueta vacíos'}), 400
    
    if label not in diseases:
        return jsonify({'error': f'Etiqueta no válida. Debe ser una de: {diseases}'}), 400
    
    try:
        # Debug info
        print(f"Recibida solicitud de entrenamiento para etiqueta: {label}")
        print(f"Nombre del archivo: {file.filename}")
        print(f"Tipo MIME: {file.content_type}")
        
        # Read file content
        file_content = file.read()
        print(f"Tamaño del archivo: {len(file_content)} bytes")
        
        # Process image
        try:
            img = Image.open(io.BytesIO(file_content))
            print(f"Formato de imagen: {img.format}, Modo: {img.mode}, Tamaño: {img.size}")
            img = img.convert('RGB')
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        except Exception as img_error:
            print(f"Error al procesar la imagen: {img_error}")
            return jsonify({'error': f'Error al procesar la imagen: {str(img_error)}'}), 400
        
        # Save image for training
        base_dir = '/tmp/dataset/train'
        os.makedirs(f'{base_dir}/{label}', exist_ok=True)
        img_count = len(os.listdir(f'{base_dir}/{label}'))
        img_path = f'{base_dir}/{label}/{img_count+1}.jpg'
        img.save(img_path)
        
        print(f"Imagen guardada en: {img_path}")
        
        # Simple incremental training if possible
        training_status = "Imagen guardada para entrenamiento"
        if model is not None:
            try:
                # We'll just acknowledge the image is saved but not actually retrain
                # to avoid memory issues on Render.com
                training_status = "Imagen guardada, pero el modelo no se reentrenará automáticamente para evitar problemas de memoria"
            except Exception as train_error:
                print(f"Error en entrenamiento incremental: {train_error}")
                training_status = "Imagen guardada, pero ocurrió un error al intentar reentrenar"
        else:
            training_status = "Imagen guardada, pero el modelo no está disponible"
        
        return jsonify({
            'status': 'success',
            'message': f'Imagen guardada para entrenamiento como {label}. {training_status}'
        })
    
    except Exception as e:
        print(f"Error en endpoint /train: {str(e)}")
        return jsonify({'error': f'Error al procesar la imagen: {str(e)}'}), 500

# Manual retraining endpoint
@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        global model
        
        # Check if we have enough data
        base_dir = '/tmp/dataset'
        if not os.path.exists(f'{base_dir}/train'):
            return jsonify({
                'status': 'error',
                'message': 'No hay suficientes datos para reentrenar el modelo'
            }), 400
        
        # Count images
        total_images = 0
        for disease in diseases:
            disease_dir = f'{base_dir}/train/{disease}'
            if os.path.exists(disease_dir):
                total_images += len(os.listdir(disease_dir))
        
        if total_images < 6:  # At least one image per class
            return jsonify({
                'status': 'error',
                'message': f'Se necesitan más imágenes para reentrenar (actual: {total_images})'
            }), 400
        
        # Create a new model
        print("Creating new model for retraining...")
        model = create_efficient_model()
        
        # Train with existing data
        train_model_with_minimal_data()
        
        # Save model
        try:
            model.save('/tmp/eye_disease_model.h5')
            print("Retrained model saved to /tmp/eye_disease_model.h5")
        except Exception as save_error:
            print(f"Error saving retrained model: {save_error}")
        
        return jsonify({
            'status': 'success',
            'message': 'Modelo reentrenado exitosamente'
        })
    
    except Exception as e:
        print(f"Error en endpoint /retrain: {str(e)}")
        return jsonify({'error': f'Error al reentrenar el modelo: {str(e)}'}), 500

# Memory cleanup endpoint
@app.route('/cleanup', methods=['POST'])
def cleanup():
    try:
        import gc
        
        # Clear TensorFlow session
        tf.keras.backend.clear_session()
        
        # Force garbage collection
        gc.collect()
        
        return jsonify({
            'status': 'success',
            'message': 'Limpieza de memoria realizada'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error durante la limpieza de memoria: {str(e)}'
        }), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    try:
        # Check if model is loaded
        model_status = "loaded" if model is not None else "not loaded"
        
        # Check if directories exist
        dirs = {
            "tmp": os.path.exists("/tmp"),
            "tmp/dataset": os.path.exists("/tmp/dataset"),
            "tmp/dataset/train": os.path.exists("/tmp/dataset/train"),
            "tmp/dataset/validation": os.path.exists("/tmp/dataset/validation")
        }
        
        # Check if model file exists
        model_file = os.path.exists("/tmp/eye_disease_model.h5")
        
        # Get memory info
        import psutil
        memory = {
            "total": psutil.virtual_memory().total / (1024 * 1024),
            "available": psutil.virtual_memory().available / (1024 * 1024),
            "used": psutil.virtual_memory().used / (1024 * 1024),
            "percent": psutil.virtual_memory().percent
        }
        
        return jsonify({
            "status": "healthy",
            "model": model_status,
            "directories": dirs,
            "model_file": model_file,
            "memory_mb": memory,
            "python_version": os.sys.version,
            "tensorflow_version": tf.__version__
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

# Initialize the model when the app starts
if __name__ == '__main__':
    print("Starting app in development mode")
    load_model()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
else:
    # This will run when Gunicorn starts the app
    print("Iniciando aplicación con Gunicorn, cargando modelo...")
    # We'll load the model on first request instead of at startup
    # to avoid memory issues during deployment
