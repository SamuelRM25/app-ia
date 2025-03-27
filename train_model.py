import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

# Definir las clases de enfermedades oculares
diseases = [
    'Normal',
    'Catarata',
    'Glaucoma',
    'Retinopatía diabética',
    'Degeneración macular',
    'Conjuntivitis'
]

def create_model(input_shape=(224, 224, 3), num_classes=len(diseases)):
    """Crear modelo CNN para clasificación de enfermedades oculares"""
    model = Sequential([
        # Primera capa convolucional
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling2D(2, 2),
        
        # Segunda capa convolucional
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        
        # Tercera capa convolucional
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        
        # Cuarta capa convolucional
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        
        # Aplanar para capas densas
        Flatten(),
        
        # Capas densas
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compilar modelo
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_dataset():
    """Preparar el conjunto de datos para entrenamiento y validación"""
    # Crear directorios si no existen
    os.makedirs('dataset/train', exist_ok=True)
    os.makedirs('dataset/validation', exist_ok=True)
    
    # Crear directorios para cada enfermedad
    for disease in diseases:
        os.makedirs(f'dataset/train/{disease}', exist_ok=True)
        os.makedirs(f'dataset/validation/{disease}', exist_ok=True)
    
    # Verificar si hay suficientes imágenes para entrenamiento
    total_images = 0
    for disease in diseases:
        if os.path.exists(f'dataset/train/{disease}'):
            total_images += len(os.listdir(f'dataset/train/{disease}'))
    
    if total_images < 30:
        print(f"Advertencia: Solo hay {total_images} imágenes para entrenamiento.")
        print("Se recomienda al menos 30 imágenes para un entrenamiento básico.")
        print("Se utilizará un conjunto de datos de prueba para demostración.")
        create_test_dataset()

def create_test_dataset():
    """Crear un conjunto de datos de prueba para demostración"""
    # Crear imágenes de prueba para cada enfermedad
    for disease in diseases:
        os.makedirs(f'dataset/train/{disease}', exist_ok=True)
        
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
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (224, 224), color=color)
            draw = ImageDraw.Draw(img)
            
            # Añadir un círculo para simular el ojo
            draw.ellipse((50, 50, 174, 174), fill=(255, 255, 255))
            draw.ellipse((80, 80, 144, 144), fill=(0, 0, 0))
            
            # Guardar imagen
            img.save(f'dataset/train/{disease}/{i+1}.jpg')
    
    # Dividir en entrenamiento y validación
    for disease in diseases:
        os.makedirs(f'dataset/validation/{disease}', exist_ok=True)
        files = os.listdir(f'dataset/train/{disease}')
        
        # Mover 2 imágenes a validación
        for i in range(2):
            if i < len(files):
                shutil.move(
                    f'dataset/train/{disease}/{files[i]}',
                    f'dataset/validation/{disease}/{files[i]}'
                )

def train_model(epochs=20, batch_size=32):
    """Entrenar el modelo con el conjunto de datos disponible"""
    # Preparar el conjunto de datos
    prepare_dataset()
    
    # Crear generadores de datos con aumento de datos
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
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        'dataset/validation',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    # Crear modelo
    model = create_model()
    
    # Callback para guardar el mejor modelo
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'eye_disease_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Entrenar modelo
    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, train_generator.samples // batch_size),
        validation_data=validation_generator,
        validation_steps=max(1, validation_generator.samples // batch_size),
        epochs=epochs,
        callbacks=[checkpoint]
    )
    
    # Guardar el modelo final
    model.save('eye_disease_model.h5')
    
    return model, history

def evaluate_model(model, history):
    """Evaluar y visualizar el rendimiento del modelo"""
    # Graficar precisión y pérdida
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Precisión de entrenamiento')
    plt.plot(val_acc, label='Precisión de validación')
    plt.legend()
    plt.title('Precisión')
    
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Pérdida de entrenamiento')
    plt.plot(val_loss, label='Pérdida de validación')
    plt.legend()
    plt.title('Pérdida')
    
    plt.savefig('training_performance.png')
    plt.show()
    
    # Imprimir resumen del modelo
    print("\nResumen del modelo:")
    model.summary()
    
    # Imprimir métricas finales
    print(f"\nPrecisión final de entrenamiento: {acc[-1]:.4f}")
    print(f"Precisión final de validación: {val_acc[-1]:.4f}")
    print(f"Pérdida final de entrenamiento: {loss[-1]:.4f}")
    print(f"Pérdida final de validación: {val_loss[-1]:.4f}")

def predict_sample_images(model):
    """Predecir algunas imágenes de muestra para verificar el modelo"""
    # Crear directorio para imágenes de prueba si no existe
    os.makedirs('test_predictions', exist_ok=True)
    
    # Cargar algunas imágenes de validación
    validation_images = []
    validation_labels = []
    
    for i, disease in enumerate(diseases):
        disease_dir = f'dataset/validation/{disease}'
        if os.path.exists(disease_dir) and len(os.listdir(disease_dir)) > 0:
            # Tomar la primera imagen de cada enfermedad
            img_path = os.path.join(disease_dir, os.listdir(disease_dir)[0])
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            validation_images.append(img_array)
            validation_labels.append(i)
    
    if not validation_images:
        print("No se encontraron imágenes de validación para probar.")
        return
    
    # Convertir a arrays numpy
    validation_images = np.array(validation_images)
    validation_labels = np.array(validation_labels)
    
    # Realizar predicciones
    predictions = model.predict(validation_images)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Mostrar resultados
    plt.figure(figsize=(15, 10))
    for i, (img, true_label, pred_label) in enumerate(zip(validation_images, validation_labels, predicted_classes)):
        plt.subplot(2, 3, i+1)
        plt.imshow(img)
        color = 'green' if true_label == pred_label else 'red'
        title = f"Real: {diseases[true_label]}\nPred: {diseases[pred_label]}"
        plt.title(title, color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('test_predictions/sample_predictions.png')
    plt.show()
    
    # Imprimir resultados
    print("\nResultados de predicción:")
    for i, (true_label, pred_label) in enumerate(zip(validation_labels, predicted_classes)):
        status = "✓" if true_label == pred_label else "✗"
        print(f"{status} Real: {diseases[true_label]}, Predicción: {diseases[pred_label]}")

def incremental_training(epochs=5, batch_size=32):
    """Realizar entrenamiento incremental si ya existe un modelo"""
    if os.path.exists('eye_disease_model.h5'):
        print("Cargando modelo existente para entrenamiento incremental...")
        model = tf.keras.models.load_model('eye_disease_model.h5')
        
        # Preparar generadores de datos
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
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            'dataset/validation',
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        # Callback para guardar el mejor modelo
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'eye_disease_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        # Entrenar modelo incrementalmente
        history = model.fit(
            train_generator,
            steps_per_epoch=max(1, train_generator.samples // batch_size),
            validation_data=validation_generator,
            validation_steps=max(1, validation_generator.samples // batch_size),
            epochs=epochs,
            callbacks=[checkpoint]
        )
        
        return model, history
    else:
        print("No se encontró un modelo existente. Realizando entrenamiento completo...")
        return train_model(epochs=epochs, batch_size=batch_size)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenar modelo de detección de enfermedades oculares')
    parser.add_argument('--incremental', action='store_true', help='Realizar entrenamiento incremental')
    parser.add_argument('--epochs', type=int, default=20, help='Número de épocas para entrenamiento')
    parser.add_argument('--batch-size', type=int, default=32, help='Tamaño del lote para entrenamiento')
    parser.add_argument('--evaluate', action='store_true', help='Evaluar modelo después del entrenamiento')
    parser.add_argument('--predict', action='store_true', help='Predecir imágenes de muestra después del entrenamiento')
    
    args = parser.parse_args()
    
    # Entrenar modelo
    if args.incremental:
        model, history = incremental_training(epochs=args.epochs, batch_size=args.batch_size)
    else:
        model, history = train_model(epochs=args.epochs, batch_size=args.batch_size)
    
    # Evaluar modelo
    if args.evaluate:
        evaluate_model(model, history)
    
    # Predecir imágenes de muestra
    if args.predict:
        predict_sample_images(model)
    
    print("Proceso de entrenamiento completado.")
