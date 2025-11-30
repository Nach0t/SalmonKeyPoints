from ultralytics import YOLO
import cv2
import os
import yaml
import numpy as np

# ============================
# CONFIGURACIÓN
# ============================
MODEL_NAME = "yolo11x-pose.pt" # Modelo base
K_FOLDS = 5
IMG_SIZE = 640
BATCH_SIZE = 6 # Baja a 4 si tienes poca VRAM

# Rutas base
BASE_IMG_DIR = "images"
BASE_LBL_DIR = "labels" # YOLO buscará aquí automáticamente si la estructura es paralela
TEST_DIR = "images/test"

# Salida de predicciones finales del TEST
OUT_LABEL_TEST = "labels/test_pred"
OUT_IMAGE_TEST = "predictions_test"

os.makedirs(OUT_LABEL_TEST, exist_ok=True)
os.makedirs(OUT_IMAGE_TEST, exist_ok=True)

# ============================
# 1. FUNCIÓN PARA CREAR YAML DINÁMICO
# ============================
def crear_yaml_fold(fold_idx, train_folds, val_fold):
    """
    Crea un archivo data_fold_X.yaml automáticamente con las rutas absolutas.
    """
    cwd = os.getcwd()
    
    train_paths = [os.path.join(cwd, BASE_IMG_DIR, f) for f in train_folds]
    val_path = os.path.join(cwd, BASE_IMG_DIR, val_fold)
    
    yaml_data = {
        'path': cwd,
        'train': train_paths,
        'val': val_path,
        'names': {0: 'salmon'},
        'kpt_shape': [4, 3]
    }
    
    yaml_filename = f"data_fold_{fold_idx}.yaml"
    with open(yaml_filename, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)
    
    return yaml_filename

# ============================
# 2. CROSS VALIDATION LOOP
# ============================
def cross_validation():
    print("\nINICIANDO CROSS-VALIDATION...\n")
    
    fold_metrics = []
    best_map = 0
    best_model_path = ""

    all_folds = [f"fold{i}" for i in range(1, K_FOLDS + 1)]

    for i in range(K_FOLDS):
        current_fold_idx = i + 1
        
        val_fold = all_folds[i]
        train_folds = [f for j, f in enumerate(all_folds) if j != i]
        
        print("\n========================================")
        print(f"FOLD {current_fold_idx}/{K_FOLDS}")
        print(f"   Train: {train_folds}")
        print(f"   Val:   {val_fold}")
        print("========================================\n")
        
        yaml_file = crear_yaml_fold(current_fold_idx, train_folds, val_fold)
        
        model = YOLO(MODEL_NAME)
        
        project_name = "runs/cv_pose"
        run_name = f"fold_{current_fold_idx}"
        
        model.train(
            data=yaml_file,
            imgsz=IMG_SIZE,
            epochs=20,
            batch=BATCH_SIZE,
            device=0,
            project=project_name,
            name=run_name,
            exist_ok=True,
            plots=True,
            verbose=False
        )
        
        metrics = model.val()
        map50_95 = metrics.pose.map
        
        print(f"Resultados Fold {current_fold_idx}: mAP50-95 = {map50_95:.4f}")
        fold_metrics.append(map50_95)
        
        if map50_95 > best_map:
            best_map = map50_95
            best_model_path = os.path.join(project_name, run_name, "weights", "best.pt")

    print("\nRESULTADOS CROSS-VALIDATION")
    for i, m in enumerate(fold_metrics):
        print(f"   Fold {i+1}: {m:.4f}")
    
    print("--------------------------")
    print(f"PROMEDIO mAP: {np.mean(fold_metrics):.4f}")
    print(f"MEJOR MODELO: {best_model_path}")
    
    return best_model_path

# ============================
# 3. PREDICCIÓN EN TEST (Auto-Labeling)
# ============================
def predecir_test(model_path):
    print(f"\nGENERANDO ETIQUETAS PARA TEST ({TEST_DIR})...\n")
    
    if not os.path.exists(model_path):
        print("Error: No se encuentra el mejor modelo.")
        return

    model = YOLO(model_path)
    imgs = [f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    
    print(f"Procesando {len(imgs)} imágenes de test...")

    for i, name in enumerate(imgs):
        img_path = os.path.join(TEST_DIR, name)
        
        results = model(img_path, device=0, conf=0.25, verbose=False)[0]
        
        if results.keypoints is None:
            continue

        kpts_norm = results.keypoints.xyn.cpu().numpy()
        if len(kpts_norm) == 0:
            continue

        txt_path = os.path.join(OUT_LABEL_TEST, name.rsplit(".", 1)[0] + ".txt")
        with open(txt_path, "w") as f:
            for kpts in kpts_norm:
                linea = "0 0.5 0.5 1.0 1.0 "
                for kx, ky in kpts:
                    vis = 2 if (kx > 0 or ky > 0) else 0
                    linea += f"{kx:.6f} {ky:.6f} {vis} "
                f.write(linea.strip() + "\n")

        img = cv2.imread(img_path)
        if img is not None:
            kpts_abs = results.keypoints.xy.cpu().numpy()
            for person in kpts_abs:
                for x, y in person:
                    if x == 0 and y == 0:
                        continue
                    cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
            cv2.imwrite(os.path.join(OUT_IMAGE_TEST, f"pred_{name}"), img)

        if i % 10 == 0:
            print(f"Progreso test: {i}/{len(imgs)}...")

    print("\nProceso completado.")
    print(f"Etiquetas TEST generadas en: {OUT_LABEL_TEST}")

# ============================
# MAIN
# ============================
if __name__ == "__main__":
    try:
        mejor_modelo = cross_validation()
        predecir_test(mejor_modelo)
    except Exception as e:
        print(f"\nERROR CRITICO: {e}")
