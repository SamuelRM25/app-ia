from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import os
import io
from PIL import Image, ImageDraw
# Remove flask-cors dependency
# from flask_cors import CORS

app = Flask(__name__)
# Instead of using CORS, we'll add CORS headers manually
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Variables globales
model = None
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

# Función para cargar el modelo
def load_model():
    global model
    try:
        # Intentar cargar el modelo si existe
        if os.path.exists('eye_disease_model.h5'):
            model = tf.keras.models.load_model('eye_disease_model.h5')
            print("Modelo cargado exitosamente")
        else:
            # Si no existe, crear un modelo básico para pruebas
            print("Modelo no encontrado, creando modelo de prueba")
            model = create_test_model()
            # Entrenar el modelo con datos de prueba
            train_test_model(model)
    except Exception as e:
        print(f"Error al cargar/entrenar el modelo: {e}")
        # Crear un modelo básico para pruebas
        model = create_test_model()

# Función para entrenar el modelo con datos de prueba
def train_test_model(model):
    import shutil
    
    # Crear directorios para datos de prueba
    os.makedirs('dataset/train', exist_ok=True)
    os.makedirs('dataset/validation', exist_ok=True)
    
    # Crear datos de prueba para cada enfermedad
    for disease in diseases:
        os.makedirs(f'dataset/train/{disease}', exist_ok=True)
        os.makedirs(f'dataset/validation/{disease}', exist_ok=True)
        
        # Crear 10 imágenes de prueba por enfermedad
        for i in range(10):
            # Crear una imagen de color sólido diferente para cada enfermedad
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
            
            # Crear imagen
            img = Image.new('RGB', (224, 224), color=color)
            draw = ImageDraw.Draw(img)
            
            # Añadir un círculo para simular el ojo
            draw.ellipse((50, 50, 174, 174), fill=(255, 255, 255))
            draw.ellipse((80, 80, 144, 144), fill=(0, 0, 0))
            
            # Guardar imagen
            img.save(f'dataset/train/{disease}/{i+1}.jpg')
    
    # Mover algunas imágenes a validación
    for disease in diseases:
        files = os.listdir(f'dataset/train/{disease}')
        
        # Mover 2 imágenes a validación
        for i in range(2):
            if i < len(files):
                shutil.move(
                    f'dataset/train/{disease}/{files[i]}',
                    f'dataset/validation/{disease}/{files[i]}'
                )
    
    # Entrenar el modelo
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Crear generadores de datos
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical'
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        'dataset/validation',
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical'
    )
    
    # Entrenar modelo
    model.fit(
        train_generator,
        steps_per_epoch=max(1, train_generator.samples // 16),
        validation_data=validation_generator,
        validation_steps=max(1, validation_generator.samples // 16),
        epochs=5
    )
    
    # Guardar el modelo
    model.save('eye_disease_model.h5')
    print("Modelo entrenado y guardado exitosamente")

# Función para crear un modelo de prueba simple
def create_test_model():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(diseases), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

@app.route('/')
def index():
    return jsonify({
        'status': 'success',
        'message': 'API de detección de enfermedades oculares funcionando correctamente'
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No se encontró imagen en la solicitud'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
    
    try:
        # Procesar la imagen
        img = Image.open(io.BytesIO(file.read()))
        img = img.convert('RGB')  # Asegurar que la imagen esté en formato RGB
        img = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Realizar predicción
        prediction = model.predict(img_array)
        disease_index = np.argmax(prediction[0])
        confidence = prediction[0][disease_index] * 100
        
        disease_name = diseases[disease_index]
        disease_info = create_disease_info()[disease_name]
        
        result = {
            'disease': disease_name,
            'confidence': f"{confidence:.2f}%",
            'description': disease_info['description'],
            'medications': disease_info['medications'],
            'treatment': disease_info['treatment']
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Error al procesar la imagen: {str(e)}'}), 500

# Add the new train endpoint
@app.route('/train', methods=['POST'])
def train():
    if 'image' not in request.files or 'label' not in request.form:
        return jsonify({'error': 'Se requiere una imagen y una etiqueta'}), 400
    
    file = request.files['image']
    label = request.form['label']
    
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
    
    if label not in diseases:
        return jsonify({'error': 'Etiqueta de enfermedad no válida'}), 400
    
    try:
        # Crear directorios si no existen
        os.makedirs(f'dataset/train/{label}', exist_ok=True)
        
        # Guardar la imagen
        img = Image.open(io.BytesIO(file.read()))
        img = img.convert('RGB')  # Asegurar que la imagen esté en formato RGB
        img = img.resize((224, 224))
        
        # Generar un nombre de archivo único
        filename = f"{label}_{len(os.listdir(f'dataset/train/{label}')) + 1}.jpg"
        img.save(f'dataset/train/{label}/{filename}')
        
        return jsonify({
            'message': f'Imagen guardada correctamente para entrenamiento como {label}',
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': f'Error al procesar la imagen: {str(e)}'}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        # Aquí implementaríamos el reentrenamiento del modelo
        # Por ahora, solo devolvemos un mensaje
        return jsonify({
            'status': 'success',
            'message': 'Función de reentrenamiento no implementada completamente'
        })
    except Exception as e:
        return jsonify({'error': f'Error al reentrenar el modelo: {str(e)}'}), 500

@app.route('/echo', methods=['POST'])
def echo():
    data = request.json
    return jsonify({
        'status': 'success',
        'message': 'Echo endpoint',
        'data': data
    })

if __name__ == '__main__':
    load_model()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
