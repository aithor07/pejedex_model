import sys
import numpy as np
from PIL import Image
import csv

# Intentar importar el runtime ligero de TFLite o el completo
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

MODEL_PATH = "inat_model.tflite"
LABELS_PATH = "labels.csv"

def cargar_etiquetas(ruta_csv):
    """Carga el mapeo del CSV de etiquetas del modelo."""
    etiquetas = []
    with open(ruta_csv, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                etiquetas.append(row[0].strip())
    return etiquetas

def clasificar_captura(imagen_path, top_k=5):
    # 1. Cargar intérprete TFLite
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Dimensiones que espera el modelo (habitualmente 224x224 o 299x299)
    _, req_height, req_width, _ = input_details[0]['shape']

    # 2. Cargar y preparar la imagen
    img = Image.open(imagen_path).convert("RGB")
    img = img.resize((req_width, req_height))
    input_data = np.array(img, dtype=np.float32)

    # Normalización si el modelo espera floats [-1, 1] o [0, 1]
    if input_details[0]['dtype'] == np.float32:
        input_data = (input_data / 127.5) - 1.0
    elif input_details[0]['dtype'] == np.uint8:
        input_data = np.array(img, dtype=np.uint8)

    input_data = np.expand_dims(input_data, axis=0)

    # 3. Ejecutar inferencia local
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # 4. Obtener predicciones
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Si son logits o enteros, calcular softmax para tener porcentajes
    exp_scores = np.exp(output_data - np.max(output_data))
    probabilidades = exp_scores / np.sum(exp_scores)

    etiquetas = cargar_etiquetas(LABELS_PATH)

    # Obtener los mejores resultados
    indices_top = np.argsort(probabilidades)[::-1][:top_k]

    print(f"\nResultados para: {imagen_path}")
    print("=" * 45)
    for i, idx in enumerate(indices_top):
        nombre = etiquetas[idx] if idx < len(etiquetas) else f"ID #{idx}"
        confianza = probabilidades[idx] * 100
        print(f"{i + 1}. {nombre:<30} | Certeza: {confianza:6.2f}%")
    print("=" * 45 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python probar_pez.py <ruta_de_tu_foto.jpg>")
    else:
        clasificar_captura(sys.argv[1])