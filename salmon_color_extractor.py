import cv2
import numpy as np
import os

# ============================
# 1. CONFIGURACIÓN
# ============================
PRED_IMG_DIR = "predictions"
PRED_TXT_DIR = "labels/val_pred"
OUTPUT_DIR = "predictions_roche_calibrado"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# 2. DICCIONARIO CALIBRADO (EXTRAÍDO DE TU FOTO)
# ============================
# Estos valores BGR (Blue, Green, Red) han sido estimados
# analizando la foto de la regla que subiste.
ROCHE_TARGETS = {
    20: np.array([115, 135, 180]), # Color extraído del cuadro 20
    21: np.array([105, 125, 190]),
    22: np.array([95,  115, 200]),
    23: np.array([85,  105, 210]),
    24: np.array([75,  95,  220]),
    25: np.array([65,  85,  225]),
    26: np.array([60,  80,  230]),
    27: np.array([55,  75,  235]),
    28: np.array([50,  70,  240]), # Naranja intenso típico
    29: np.array([45,  65,  235]),
    30: np.array([40,  60,  230]),
    31: np.array([40,  55,  225]),
    32: np.array([40,  50,  220]),
    33: np.array([35,  45,  215]),
    34: np.array([35,  40,  210])  # Rojo oscuro extraído del cuadro 34
}

# Pre-convertimos estos objetivos a espacio de color LAB.
# LAB es mucho mejor que RGB para medir diferencias de color como el ojo humano.
LAB_TARGETS = {}
for score, bgr in ROCHE_TARGETS.items():
    pixel = np.uint8([[bgr]])
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2Lab)[0][0].astype(np.int32)
    LAB_TARGETS[score] = lab

# ============================
# 3. FUNCIÓN DE ESTIMACIÓN (VECINO MÁS CERCANO)
# ============================
def estimate_roche_lab(bgr_pixel):
    """
    Convierte el pixel del salmón a LAB y busca cuál es el 
    cuadro de la regla que tiene la menor distancia de color.
    """
    # Convertir pixel de entrada a LAB
    pixel_in = np.uint8([[bgr_pixel]])
    lab_in = cv2.cvtColor(pixel_in, cv2.COLOR_BGR2Lab)[0][0].astype(np.int32)

    min_dist = float('inf')
    best_roche = 20

    # Pesos para la distancia (Luminosidad importa menos, Color importa más)
    # Esto ayuda a que una sombra no cambie el resultado drásticamente.
    w_L = 0.6 
    w_A = 1.0
    w_B = 1.0

    for score, lab_target in LAB_TARGETS.items():
        # Distancia Euclidiana Ponderada
        dL = (lab_in[0] - lab_target[0]) * w_L
        dA = (lab_in[1] - lab_target[1]) * w_A
        dB = (lab_in[2] - lab_target[2]) * w_B
        
        dist = np.sqrt(dL**2 + dA**2 + dB**2)

        if dist < min_dist:
            min_dist = dist
            best_roche = score

    return best_roche

# ============================
# 4. PROCESAMIENTO
# ============================
print(f"Iniciando estimación calibrada con regla SalmoFan...\n")

for txt_file in os.listdir(PRED_TXT_DIR):
    if not txt_file.endswith(".txt"): continue

    txt_path = os.path.join(PRED_TXT_DIR, txt_file)
    
    # Leer archivo TXT
    with open(txt_path, "r") as f:
        lines = f.readlines()
        if not lines: continue
        parts = lines[0].split()

    if len(parts) < 6: continue

    # Buscar imagen asociada
    img_name = txt_file.replace(".txt", ".png")
    if not os.path.exists(os.path.join(PRED_IMG_DIR, "pred_" + img_name)):
         img_name = txt_file.replace(".txt", ".jpg")

    img_path = os.path.join(PRED_IMG_DIR, "pred_" + img_name)
    if not os.path.exists(img_path): 
        print(f"Falta imagen: {img_name}")
        continue

    img = cv2.imread(img_path)
    if img is None: continue
    h, w = img.shape[:2]

    kpts = parts[5:]
    roche_values = []

    # Iterar sobre los puntos clave detectados por YOLO
    for i in range(0, len(kpts), 3):
        x_norm, y_norm, vis = float(kpts[i]), float(kpts[i+1]), float(kpts[i+2])
        
        # Filtro suave: solo ignorar si YOLO dice que no es visible
        if vis < 0.2 or x_norm <= 0: continue
        
        xi, yi = int(x_norm * w), int(y_norm * h)

        # Extraer muestra de color (promedio 5x5 pixeles)
        patch = img[max(0, yi-2):yi+3, max(0, xi-2):xi+3]
        if patch.size == 0: continue
        
        bgr_avg = patch.mean(axis=(0, 1))

        # ESTIMAR ROCHE USANDO CALIBRACIÓN
        score = estimate_roche_lab(bgr_avg)
        roche_values.append(score)

        # DIBUJAR
        # Usar el color exacto de la regla para el círculo
        target_color = ROCHE_TARGETS.get(score, [255,255,255]).tolist()
        
        # Círculo con el color detectado
        cv2.circle(img, (xi, yi), 6, target_color, -1)
        cv2.circle(img, (xi, yi), 7, (0,0,0), 1) # Borde negro

        # Texto con el número
        cv2.putText(img, str(score), (xi + 10, yi - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, str(score), (xi + 10, yi - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Calcular promedio del filete
    if roche_values:
        avg_roche = int(round(np.mean(roche_values)))
        print(f"✓ {img_name}: Roche Promedio = {avg_roche}")
        
        # Escribir resultado grande
        cv2.rectangle(img, (10, 10), (250, 70), (0,0,0), -1)
        cv2.putText(img, f"ROCHE: {avg_roche}", (20, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    else:
        print(f"⚠ {img_name}: No se detectaron puntos válidos.")

    out_path = os.path.join(OUTPUT_DIR, img_name)
    cv2.imwrite(out_path, img)

print(f"\n🎉 COMPLETADO. Resultados guardados en: {OUTPUT_DIR}")