
# 📘 Guía de Ejecución — `train_model.py`

Este documento explica paso a paso cómo ejecutar el script **`train_model.py`**, cuyo objetivo es entrenar y comparar dos modelos de regresión para el TP Properati:

- **LinearRegression**
- **RandomForestRegressor**

El script también guarda predicciones y métricas en la base SQLite.

---

## ✅ 1. Requisitos previos

Antes de ejecutar el entrenamiento, asegurate de tener listos los siguientes elementos:

### 📌 Archivos necesarios dentro de `data/artifacts/`

| Archivo | Generado por | ¿Para qué sirve? |
|--------|--------------|------------------|
| `X_preprocessed.npz` o `.npy` | preprocessing.py | Matriz de features lista para modelado |
| `y.npy` | preprocessing.py | Vector objetivo (precio) |
| `database.db` | load_to_sqlite.py | Base donde se guardarán métricas y predicciones |
| `feature_names.json` | preprocessing.py | Nombres de columnas transformadas |

Ejemplo de estructura esperada:

```
tppa/
 ├─ Scripts/
 │   └─ train_model.py
 ├─ data/
 │   └─ artifacts/
 │       ├─ X_preprocessed.npz
 │       ├─ y.npy
 │       ├─ database.db
 │       └─ feature_names.json
```

---

## ✅ 2. Activar entorno virtual

En PowerShell, desde la raíz del proyecto:

```powershell
.	ppa\Scripts\Activate.ps1
```

---

## ✅ 3. Ejecutar el script

Desde la raíz del proyecto:

```powershell
python .\Scripts\train_model.py --db_path .\data\artifacts\database.db --artifacts_dir .\data\artifacts --max_samples 20000
```

### 📌 Parámetros disponibles

| Parámetro | Descripción | Default |
|----------|-------------|---------|
| `--db_path` | Ruta a `database.db` | `data/artifacts/database.db` |
| `--artifacts_dir` | Carpeta donde viven X e y | `data/artifacts` |
| `--test_size` | Proporción para test | `0.2` |
| `--max_samples` | Filas máximas a usar para evitar problemas de RAM | `150000` |
| `--random_state` | Semilla | `42` |

Ejemplo con parámetros personalizados:

```powershell
python .\Scripts	rain_model.py --test_size 0.25 --max_samples 100000
```

---

## ✅ 4. ¿Qué ocurre durante la ejecución?

El script:

1. **Carga X e y desde `data/artifacts/`.**
2. **Submuestrea** (si X es muy grande) para manejar memoria.
3. Realiza `train_test_split`.
4. Entrena:
   - LinearRegression  
   - RandomForestRegressor
5. Calcula:
   - RMSE  
   - MAE  
   - R²  
6. Inserta resultados en **SQLite**:

### Tablas creadas/llenadas

| Tabla | Contenido |
|-------|-----------|
| `model_results` | Predicciones del **set de test**, por fila |
| `model_metrics` | Métricas por modelo y split (train/test) |

Ejemplo de registros en `model_metrics`:

| model_name | split | rmse | mae | r2 |
|------------|--------|------|------|------|
| LinearRegression | test | 82000 | 54000 | 0.62 |
| RandomForestRegressor | test | 69000 | 48000 | 0.71 |

---

## ✅ 5. Cómo verificar los resultados

### Ver métricas en SQLite

Abrí DB Browser for SQLite → `database.db` → pestaña **Browse Data** → tabla `model_metrics`.

Consulta SQL rápida:

```sql
SELECT model_name, split, rmse, mae, r2
FROM model_metrics
ORDER BY split, rmse;
```

### Ver predicciones

```sql
SELECT *
FROM model_results
WHERE model_name = 'RandomForestRegressor'
LIMIT 20;
```

---

## 📊 6. Interpretación esperada

Generalmente:

- **RandomForestRegressor** suele obtener mejores métricas que LinearRegression.
- La comparación se basa en:
  - RMSE más bajo
  - MAE más bajo
  - R² más alto

Esto cumple con el requerimiento del TP: **comparar al menos dos modelos de regresión**.

---

## 🎉 ¡Listo!

Tu pipeline completo ahora incluye:

1. Preprocesamiento  
2. Carga en SQLite  
3. EDA  
4. Entrenamiento + comparación de modelos  
5. Métricas guardadas de forma reproducible  

Si querés, puedo generarte también:

- Un **train_model_v2.py** con barra de progreso, logs o guardar el modelo entrenado.
- Un **script evaluate_model.py** para comparar modelos automáticamente.
- Un **informe en DOCX** con gráficos, tabla de métricas y conclusiones.

Solo pedímelo 🙂
