# Proyecto Final: Regresión de Precios de Propiedades (Properati)

Este proyecto fue desarrollado para la materia *Programación Avanzada para Ciencia de Datos*. El objetivo es construir un pipeline de Machine Learning para predecir el precio de propiedades en Argentina, utilizando el dataset de Properati.

El flujo de trabajo incluye la limpieza y preprocesamiento de datos, análisis exploratorio, almacenamiento en una base de datos, entrenamiento y comparación de modelos de regresión.

## 1\. Objetivos del Proyecto

Los requisitos principales con los que cuenta nuestro proyecto son:

  * **Construcción del Modelo:**
      * Utilizamos `Pipelines` de Scikit-Learn para el preprocesamiento automatizado.
      * Comparamos el desempeño de al menos dos algoritmos de regresión (ej. `LinearRegression` y `RandomForestRegressor`).
      * Evaluamos los modelos utilizando métricas clave: RMSE, MAE y R².
  * **Almacenamiento de Datos:**
      * Tratamos de persistir tanto los datos de entrada como los resultados del modelo en una base de datos (usamos SQLite).
      * La base de datos contiene tablas para `input_data`, `model_results` y la `model_config`.
  * **Visualización:**
      * Realizamos un Análisis Exploratorio de Datos (EDA) para entender las variables, usando `matplotlib` y `seaborn`.
 

## 2\. Herramientas

  * **Lenguaje:** Python 3.7+
  * **Análisis y Modelado:** Pandas, Scikit-learn
  * **Visualización:** Matplotlib, Seaborn
  * **Base de Datos:** SQLite
  * **Control de Versiones:** Git, GitHub
  * **Manejo de Archivos Pesados:** Git LFS (Large File Storage)

## 3\. Instalación y Ejecución


Usamos **Git LFS** para manejar archivos de gran tamaño (como el `.csv` y la base de datos `.db`).

**Primero, instalamos Git LFS** en el sistema. Luego, pudimos clonar el repositorio:

git lfs install

# Clonación del repositorio
git clone https://github.com/SheilaAguirreRivas/TPFinalProgAvanzadaCD.git

### Pasos para ejecutar el proyecto

1.  **Navegamos a la carpeta del proyecto:**

    cd TPFinalProgAvanzadaCD

2.  **Creamos y activamos un entorno virtual:**

    python -m venv venv

    # Activamos en PowerShell
    .\venv\Scripts\activate


3.  **Instalamos las dependencias:**

    pip install -r requirements.txt

4.  **Ejecutamos el Pipeline:**

      * **Paso 1 - Preprocesamiento:** Este script creó los artefactos (ej. `preprocessor.joblib`, `X_preprocessed_aligned.npz`) en la carpeta `/artifacts`.
       
      * **Paso 2 - Cargamos la Base de Datos:** Este script tomó los artefactos y el CSV para crear y llenar la base de datos `data/artifacts/database.db`.
      
      * **Paso 3 - Entrenamiento:** Este script leyó los datos de la base de datos y entrenó los modelos y guardó las métricas en la tabla `model_results` de la base de datos.

      * **Paso 4 - Exploración:** Abrimos y ejecutamos el notebook `eda_sqlite.ipynb` para ver el análisis exploratorio.



## 4. Metodología

### 4.1. Análisis Exploratorio (EDA)

Se realizó un EDA (`eda_sqlite.ipynb`) para entender la distribución de los datos. El análisis se hizo leyendo los datos directamente desde la tabla `input_data` de nuestra base de datos SQLite.

### 4.2. Preprocesamiento (Pipeline)

El script `preprocessing.py` es el pipeline automatizado. Usamos un `ColumnTransformer` de Scikit-learn para:

1.  **Valores faltantes:** Aplicar `SimpleImputer` (ej. estrategia "mediana" para numéricos y "moda" para categóricos).
2.  **Estandarizar variables numéricas:** Aplicar `StandardScaler` (ej. a `surface_total`).
3.  **Codificar variables categóricas:** Aplicar `OneHotEncoder` (ej. a `property_type`).

Este pipeline se guardó como `artifacts/preprocessor.joblib` para ser reutilizado.

### 4.3. Diseño de la Base de Datos (SQLite)

El script `load_sqlite.py` genera la base de datos `database.db` con el siguiente esquema:

* **`input_data`:** Almacenando los datos crudos del `properati_clean.csv` (features y target). Usada por el EDA.
* **`preprocessed_data`:** Almacena la matriz de features preprocesada en formato "disperso" (row_idx, feature_name, value).
* **`model_config`:** Tabla de configuración. Registra y guarda el `run_id`, los tipos de cambio usados y los paths a los artefactos.
* **`model_results`:** Tabla vacía, para a recibir los resultados del script `train_model.py`.


## 5. Autores (Equipo de Trabajo)

* Sheila Aguirre (`saguirre`)
* Claudia Medina (`cmedina`)
* Carolina Pereyra (`cpereyra`)
* Edith Cisneros (`ecisneros`)
