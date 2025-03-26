from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import os
import io
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Habilitar CORS para permitir solicitudes desde el frontend

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
# Modificar la función load_model para entrenar un modelo si no existe
def load_model():
    global model
    try:
        # Intentar cargar el modelo si existe
        if os.path.exists('eye_disease_model.h5'):
            model = tf.keras.models.load_model('eye_disease_model.h5')
            print("Modelo cargado exitosamente")
        else:
            # Si no existe, importar y ejecutar el entrenamiento
            print("Modelo no encontrado, creando modelo de entrenamiento...")
            from train_model import train_model
            model, _ = train_model(epochs=5, batch_size=16)  # Entrenamiento rápido
            print("Modelo entrenado y guardado exitosamente")
    except Exception as e:
        print(f"Error al cargar/entrenar el modelo: {e}")
        # Crear un modelo básico para pruebas
        model = create_test_model()

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
        # Procesar la imagen
        img = Image.open(io.BytesIO(file.read()))
        img = img.convert('RGB')
        img = img.resize((224, 224))
        
        # Guardar la imagen para entrenamiento
        os.makedirs(f'dataset/train/{label}', exist_ok=True)
        img_count = len(os.listdir(f'dataset/train/{label}'))
        img.save(f'dataset/train/{label}/{img_count+1}.jpg')
        
        # Aquí se podría implementar un entrenamiento incremental
        # Por ahora, solo devolvemos un mensaje de éxito
        
        return jsonify({
            'status': 'success',
            'message': f'Imagen guardada para entrenamiento como {label}'
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