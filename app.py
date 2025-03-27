from flask import Flask, request, jsonify, render_template, Response
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
    response.headers.add('Access-Control-Max-Age', '3600')
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
# Modify the load_model function to be more robust and add global model declaration
def load_model():
    global model
    print("Starting model loading process...")
    try:
        # Intentar cargar el modelo si existe
        if os.path.exists('eye_disease_model.h5'):
            print("Intentando cargar modelo existente...")
            model = tf.keras.models.load_model('eye_disease_model.h5')
            print("Modelo cargado exitosamente")
        else:
            print("Modelo no encontrado en ruta principal, buscando en /tmp...")
            # Intentar cargar desde /tmp si existe
            if os.path.exists('/tmp/eye_disease_model.h5'):
                model = tf.keras.models.load_model('/tmp/eye_disease_model.h5')
                print("Modelo cargado desde /tmp exitosamente")
            else:
                # Si no existe, crear un modelo básico para pruebas
                print("Modelo no encontrado, creando modelo de prueba")
                model = create_test_model()
                # Entrenar el modelo con datos de prueba
                train_test_model(model)
                # Guardar en /tmp para futuros usos
                model.save('/tmp/eye_disease_model.h5')
                print("Modelo de prueba creado, entrenado y guardado en /tmp")
        
        # Verificar que el modelo se haya cargado correctamente
        if model is None:
            raise Exception("El modelo sigue siendo None después de intentar cargarlo")
        else:
            print(f"Modelo cargado correctamente: {type(model)}")
            
    except Exception as e:
        print(f"Error al cargar/entrenar el modelo: {e}")
        # Crear un modelo básico para pruebas
        try:
            print("Creando modelo de respaldo después de error...")
            model = create_test_model()
            # Intentar entrenar con datos mínimos
            try:
                train_test_model(model)
                # Guardar en /tmp
                model.save('/tmp/eye_disease_model.h5')
                print("Modelo de respaldo creado y guardado en /tmp")
            except Exception as train_error:
                print(f"Error al entrenar modelo de respaldo: {train_error}")
        except Exception as model_error:
            print(f"Error crítico al crear modelo de respaldo: {model_error}")
            # En este punto, model seguirá siendo None

# Función para entrenar el modelo con datos de prueba
def train_test_model(model):
    try:
        import shutil
        
        # Usar /tmp para asegurar permisos de escritura
        base_dir = '/tmp/dataset'
        
        # Crear directorios para datos de prueba
        os.makedirs(f'{base_dir}/train', exist_ok=True)
        os.makedirs(f'{base_dir}/validation', exist_ok=True)
        
        # Crear datos de prueba para cada enfermedad
        for disease in diseases:
            os.makedirs(f'{base_dir}/train/{disease}', exist_ok=True)
            os.makedirs(f'{base_dir}/validation/{disease}', exist_ok=True)
            
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
                
                # Guardar imagen en /tmp
                img.save(f'{base_dir}/train/{disease}/{i+1}.jpg')
        
        # Mover algunas imágenes a validación
        for disease in diseases:
            files = os.listdir(f'{base_dir}/train/{disease}')
            
            # Mover 2 imágenes a validación
            for i in range(2):
                if i < len(files):
                    shutil.move(
                        f'{base_dir}/train/{disease}/{files[i]}',
                        f'{base_dir}/validation/{disease}/{files[i]}'
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
            f'{base_dir}/train',
            target_size=(224, 224),
            batch_size=16,
            class_mode='categorical'
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            f'{base_dir}/validation',
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
        
        # Guardar el modelo en /tmp
        model.save('/tmp/eye_disease_model.h5')
        print("Modelo entrenado y guardado exitosamente en /tmp")
        
        # Intentar guardar también en la ubicación principal
        try:
            model.save('eye_disease_model.h5')
            print("Modelo también guardado en ubicación principal")
        except Exception as e:
            print(f"No se pudo guardar en ubicación principal: {e}")
    
    except Exception as e:
        print(f"Error en train_test_model: {e}")
        raise

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

# Modify the predict endpoint to handle None model
@app.route('/predict', methods=['POST'])
def predict():
    print("Received prediction request")
    
    # Check if model is loaded
    global model
    if model is None:
        print("Model is None, attempting to load it again")
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
        img = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Make prediction
        print("Making prediction...")
        prediction = model.predict(img_array)
        disease_index = np.argmax(prediction[0])
        confidence = prediction[0][disease_index] * 100
        
        disease_name = diseases[disease_index]
        disease_info = create_disease_info()[disease_name]
        
        print(f"Prediction result: {disease_name} with {confidence:.2f}% confidence")
        
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
        # Añadir información de depuración
        print(f"Recibida solicitud de entrenamiento para etiqueta: {label}")
        print(f"Nombre del archivo: {file.filename}")
        print(f"Tipo MIME: {file.content_type}")
        
        # Leer el contenido del archivo una sola vez y guardarlo en memoria
        file_content = file.read()
        print(f"Tamaño del archivo: {len(file_content)} bytes")
        
        # Procesar la imagen desde el contenido en memoria
        try:
            img = Image.open(io.BytesIO(file_content))
            print(f"Formato de imagen: {img.format}, Modo: {img.mode}, Tamaño: {img.size}")
            img = img.convert('RGB')
            img = img.resize((224, 224))
        except Exception as img_error:
            print(f"Error al procesar la imagen: {img_error}")
            # Intentar guardar el archivo original para diagnóstico
            try:
                with open('/tmp/debug_image.bin', 'wb') as f:
                    f.write(file_content)
                print("Archivo de depuración guardado en /tmp/debug_image.bin")
            except:
                pass
            raise
        
        # Verificar si el directorio es escribible
        base_dir = os.path.abspath('dataset/train')
        if not os.access(os.path.dirname(base_dir), os.W_OK):
            # Si no podemos escribir, almacenar en /tmp que suele ser escribible en entornos cloud
            base_dir = '/tmp/dataset/train'
            
        # Guardar la imagen para entrenamiento
        os.makedirs(f'{base_dir}/{label}', exist_ok=True)
        img_count = len(os.listdir(f'{base_dir}/{label}'))
        img_path = f'{base_dir}/{label}/{img_count+1}.jpg'
        img.save(img_path)
        
        # Registrar la operación
        print(f"Imagen guardada en: {img_path}")
        
        # Implementar entrenamiento incremental básico si estamos en un entorno que lo permite
        try:
            # Intentar reentrenar el modelo con la nueva imagen
            if model is not None:
                # Crear un generador de datos simple para la nueva imagen
                from tensorflow.keras.preprocessing.image import ImageDataGenerator
                train_datagen = ImageDataGenerator(rescale=1./255)
                
                # Entrenar con una sola época para incorporar la nueva imagen
                if os.path.exists(base_dir) and os.listdir(base_dir):
                    print("Realizando entrenamiento incremental...")
                    train_generator = train_datagen.flow_from_directory(
                        os.path.dirname(base_dir),
                        target_size=(224, 224),
                        batch_size=1,
                        class_mode='categorical'
                    )
                    
                    model.fit(
                        train_generator,
                        steps_per_epoch=1,
                        epochs=1,
                        verbose=0
                    )
                    
                    # Intentar guardar el modelo actualizado
                    try:
                        model.save('eye_disease_model.h5')
                        print("Modelo actualizado y guardado")
                        training_status = "Modelo actualizado con la nueva imagen"
                    except Exception as save_error:
                        print(f"No se pudo guardar el modelo: {save_error}")
                        training_status = "Imagen guardada, pero no se pudo actualizar el modelo"
                else:
                    training_status = "Imagen guardada, pero no hay suficientes datos para reentrenar"
            else:
                training_status = "Imagen guardada, pero el modelo no está disponible para reentrenar"
        except Exception as train_error:
            print(f"Error en entrenamiento incremental: {train_error}")
            training_status = "Imagen guardada, pero ocurrió un error al intentar reentrenar"
        
        return jsonify({
            'status': 'success',
            'message': f'Imagen guardada para entrenamiento como {label}. {training_status}'
        })
    
    except Exception as e:
        print(f"Error en endpoint /train: {str(e)}")
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

# Añadir un endpoint para obtener estadísticas del modelo
@app.route('/model_stats', methods=['GET'])
def model_stats():
    try:
        # Contar imágenes por categoría
        categories = {}
        base_dir = 'dataset/train'
        
        # Verificar si el directorio principal existe
        if not os.path.exists(base_dir):
            # Intentar con el directorio temporal
            base_dir = '/tmp/dataset/train'
            if not os.path.exists(base_dir):
                return jsonify({
                    'categories': {disease: 0 for disease in diseases},
                    'accuracy': 0.0,
                    'loss': 0.0,
                    'history': []
                })
        
        # Contar imágenes por categoría
        for disease in diseases:
            disease_dir = f'{base_dir}/{disease}'
            if os.path.exists(disease_dir):
                categories[disease] = len(os.listdir(disease_dir))
            else:
                categories[disease] = 0
        
        # Obtener métricas del modelo si está disponible
        accuracy = 0.0
        loss = 0.0
        if model is not None:
            # Intentar evaluar el modelo con datos de validación si existen
            validation_dir = 'dataset/validation'
            if os.path.exists(validation_dir) and any(os.listdir(validation_dir)):
                try:
                    from tensorflow.keras.preprocessing.image import ImageDataGenerator
                    validation_datagen = ImageDataGenerator(rescale=1./255)
                    validation_generator = validation_datagen.flow_from_directory(
                        validation_dir,
                        target_size=(224, 224),
                        batch_size=16,
                        class_mode='categorical'
                    )
                    
                    if validation_generator.samples > 0:
                        evaluation = model.evaluate(validation_generator)
                        loss = float(evaluation[0])
                        accuracy = float(evaluation[1]) * 100
                except Exception as eval_error:
                    print(f"Error al evaluar el modelo: {eval_error}")
        
        # Historial de entrenamiento (simulado por ahora)
        history = [
            {"date": "2025-03-27", "images": sum(categories.values()), "category": "Todas"}
        ]
        
        return jsonify({
            'categories': categories,
            'accuracy': accuracy,
            'loss': loss,
            'history': history
        })
    
    except Exception as e:
        print(f"Error en endpoint /model_stats: {str(e)}")
        return jsonify({'error': f'Error al obtener estadísticas: {str(e)}'}), 500

# Añadir un endpoint para entrenamiento automático
@app.route('/auto_train', methods=['POST'])
def auto_train():
    try:
        data = request.json
        if not data or 'disease' not in data or 'count' not in data:
            return jsonify({'error': 'Se requieren los campos disease y count'}), 400
        
        disease = data['disease']
        count = int(data['count'])
        source = data.get('source', 'medical')
        
        if disease not in diseases:
            return jsonify({'error': f'Enfermedad no válida. Debe ser una de: {diseases}'}), 400
        
        if count < 1 or count > 50:
            return jsonify({'error': 'El número de imágenes debe estar entre 1 y 50'}), 400
        
        # Iniciar proceso de entrenamiento automático (simulado)
        # En un entorno real, esto podría ser un proceso en segundo plano
        
        # Verificar si el directorio es escribible
        base_dir = os.path.abspath('dataset/train')
        if not os.access(os.path.dirname(base_dir), os.W_OK):
            # Si no podemos escribir, almacenar en /tmp que suele ser escribible en entornos cloud
            base_dir = '/tmp/dataset/train'
        
        os.makedirs(f'{base_dir}/{disease}', exist_ok=True)
        
        # Simular la descarga y procesamiento de imágenes
        # En un entorno real, aquí se implementaría la descarga real de imágenes
        for i in range(count):
            # Crear una imagen sintética para simular
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
            
            # Crear imagen con variaciones para simular diversidad
            import random
            color_variation = (
                random.randint(-20, 20),
                random.randint(-20, 20),
                random.randint(-20, 20)
            )
            
            final_color = tuple(max(0, min(255, c + v)) for c, v in zip(color, color_variation))
            
            img = Image.new('RGB', (224, 224), color=final_color)
            draw = ImageDraw.Draw(img)
            
            # Añadir un círculo para simular el ojo con variaciones
            center_variation = (random.randint(-10, 10), random.randint(-10, 10))
            size_variation = random.randint(-10, 10)
            
            draw.ellipse((
                50 + center_variation[0], 
                50 + center_variation[1], 
                174 + center_variation[0], 
                174 + center_variation[1]
            ), fill=(255, 255, 255))
            
            draw.ellipse((
                80 + center_variation[0], 
                80 + center_variation[1], 
                144 + center_variation[0], 
                144 + center_variation[1]
            ), fill=(0, 0, 0))
            
            # Guardar imagen
            img_count = len(os.listdir(f'{base_dir}/{disease}'))
            img.save(f'{base_dir}/{disease}/auto_{img_count+1}.jpg')
        
        # Intentar reentrenar el modelo con las nuevas imágenes
        try:
            if model is not None:
                from tensorflow.keras.preprocessing.image import ImageDataGenerator
                
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
                
                train_generator = train_datagen.flow_from_directory(
                    os.path.dirname(base_dir),
                    target_size=(224, 224),
                    batch_size=16,
                    class_mode='categorical'
                )
                
                # Entrenar con pocas épocas para ser rápido
                model.fit(
                    train_generator,
                    steps_per_epoch=max(1, train_generator.samples // 16),
                    epochs=3,
                    verbose=1
                )
                
                # Intentar guardar el modelo actualizado
                try:
                    model.save('eye_disease_model.h5')
                    print("Modelo actualizado y guardado después del entrenamiento automático")
                except Exception as save_error:
                    print(f"No se pudo guardar el modelo: {save_error}")
        except Exception as train_error:
            print(f"Error en entrenamiento automático: {train_error}")
        
        return jsonify({
            'status': 'success',
            'message': f'Proceso de entrenamiento automático iniciado para {disease} con {count} imágenes'
        })
    
    except Exception as e:
        print(f"Error en endpoint /auto_train: {str(e)}")
        return jsonify({'error': f'Error en entrenamiento automático: {str(e)}'}), 500

# Añadir un endpoint para simular progreso de entrenamiento
@app.route('/train_progress', methods=['GET'])
def train_progress():
    def generate():
        import time
        import json
        
        # Simular progreso de entrenamiento
        for i in range(0, 101, 10):
            time.sleep(1)  # Simular trabajo
            
            if i == 0:
                status = "Iniciando descarga de imágenes..."
            elif i == 20:
                status = "Descarga completada. Procesando imágenes..."
            elif i == 40:
                status = "Preparando datos para entrenamiento..."
            elif i == 60:
                status = "Entrenando modelo..."
            elif i == 80:
                status = "Evaluando resultados..."
            else:
                status = "Completado"
            
            data = {
                'progress': i,
                'status': status
            }
            
            yield f"data: {json.dumps(data)}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/health', methods=['GET'])
def health_check():
    try:
        # Check if model is loaded
        model_status = "loaded" if model is not None else "not loaded"
        
        # Check if directories exist
        dirs = {
            "dataset": os.path.exists("dataset"),
            "dataset/train": os.path.exists("dataset/train"),
            "dataset/validation": os.path.exists("dataset/validation"),
            "tmp": os.path.exists("/tmp"),
            "tmp/dataset": os.path.exists("/tmp/dataset")
        }
        
        # Check if model file exists
        model_file = os.path.exists("eye_disease_model.h5")
        
        return jsonify({
            "status": "healthy",
            "model": model_status,
            "directories": dirs,
            "model_file": model_file,
            "python_version": os.sys.version,
            "tensorflow_version": tf.__version__
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

# Ensure model is loaded when the app starts
if __name__ == '__main__':
    load_model()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
else:
    # This will run when Gunicorn starts the app
    print("Iniciando aplicación con Gunicorn, cargando modelo...")
    # Force model loading at import time
    load_model()
    
    # Add a verification step
    if model is None:
        print("WARNING: Model is still None after loading attempt!")
    else:
        print(f"Model successfully loaded: {type(model)}")

# Add a new endpoint to force model reloading
@app.route('/reload_model', methods=['POST'])
def reload_model():
    try:
        global model
        print("Forcing model reload...")
        
        # Attempt to clear any existing model from memory
        if model is not None:
            del model
            import gc
            gc.collect()
            model = None
        
        # Load the model again
        load_model()
        
        if model is None:
            return jsonify({
                'status': 'error',
                'message': 'No se pudo cargar el modelo después de intentar recargarlo'
            }), 500
        else:
            return jsonify({
                'status': 'success',
                'message': 'Modelo recargado exitosamente',
                'model_type': str(type(model))
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error al recargar el modelo: {str(e)}'
        }), 500
