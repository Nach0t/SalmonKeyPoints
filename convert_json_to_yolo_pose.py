import json
import os

# ==== CONFIGURACIÓN ====
JSON_PATH = "project-47-at-2025-11-30-00-22-88dcc9ea.json"
IMAGES_DIR = "images/train"         # carpeta donde están tus imágenes (train/val)
OUT_LABELS_DIR = "labels/train"     # carpeta donde se guardarán los TXT

# Orden EXACTO de keypoints
KEYPOINT_ORDER = ["Cola", "Lomo", "Centro", "Belly"]

# Crear carpetas si no existen
os.makedirs(OUT_LABELS_DIR, exist_ok=True)

# Cargar JSON
with open(JSON_PATH, "r") as f:
    data = json.load(f)

print(f"Total anotaciones: {len(data)}")

for item in data:
    img_path = item["data"]["img"].split("/")[-1]  # nombre del archivo
    img_full_path = os.path.join(IMAGES_DIR, img_path)

    if not os.path.exists(img_full_path):
        print(f"⚠ Imagen no encontrada: {img_full_path}")
        continue

    # Crear .txt
    txt_path = os.path.join(OUT_LABELS_DIR, img_path.replace(".png", ".txt"))

    # SOLO hay una anotación por tarea
    ann = item["annotations"][0]["result"]

    # Imagen completa = bounding box completo
    box_x = 0.5
    box_y = 0.5
    box_w = 1.0
    box_h = 1.0

    # preparar dict de keypoints
    kps = {k: (0, 0, 0) for k in KEYPOINT_ORDER}

    for kp in ann:
        if kp["type"] != "keypointlabels":
            continue

        label = kp["value"]["keypointlabels"][0]

        if label not in KEYPOINT_ORDER:
            continue

        # Label Studio entrega x,y normalizados %
        x = kp["value"]["x"] / 100
        y = kp["value"]["y"] / 100

        # Visible = 2 (visible y presente)
        kps[label] = (x, y, 2)

    # Crear línea YOLO-Pose
    line = "0 "  # clase única (salmón)

    # bounding box
    line += f"{box_x} {box_y} {box_w} {box_h} "

    # keypoints en orden exacto
    for k in KEYPOINT_ORDER:
        x, y, v = kps[k]
        line += f"{x} {y} {v} "

    # guardar
    with open(txt_path, "w") as f:
        f.write(line.strip())

    print("✓ Label generado:", txt_path)

print("🔥 Conversión completa. Tus TXT ahora están correctos.")
