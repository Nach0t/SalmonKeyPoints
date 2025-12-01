# SalmonKeyPoints

Proyecto para detección de puntos anatómicos (keypoints) en salmón utilizando YOLO11 Pose

---

## 1. Preparación del dataset

Descargar y descomprimir `imagenes.zip` en la **raíz del repositorio**.

Debe quedar así:

```
SalmonKeyPoints/
│
├── images/
│   ├── fold1/
│   ├── fold2/
│   ├── fold3/
│   ├── fold4/
│   ├── fold5/
│   └── test/
│
└── labels/
    ├── fold1/
    ├── fold2/
    ├── fold3/
    ├── fold4/
    ├── fold5/
```

No debe quedar dentro de otra carpeta como:

```
SalmonKeyPoints/imagenes/images/...
```

---

## 2. Crear y activar entorno virtual

### Windows (PowerShell o CMD)

```
python -m venv .venv
.venv\Scripts\activate
```

### Linux / WSL

```
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Instalar dependencias

```
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Abrir Jupyter Lab

```
jupyter lab
```

Luego abrir:

```
yolo_cv_pose.ipynb
```

---

## 5. Ejecución del proyecto

El notebook realiza:

- Validación cruzada (K-Fold)
- Entrenamiento de pose
- Selección automática del mejor modelo
- Generación de predicciones en test
- Exportación de keypoints
---

