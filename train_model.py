# Scripts/train_model.py
# -*- coding: utf-8 -*-
"""
Entrenamiento y evaluación de modelos de regresión para el TP Properati
========================================================================

Este script:
1. Se conecta a la base SQLite (database.db).
2. Lee los datos preprocesados desde los archivos X_preprocessed.* e y.npy.
3. (Opcional) Muestra una cantidad máxima de filas para evitar problemas de memoria.
4. Divide en train/test.
5. Entrena dos modelos de regresión:
    - LinearRegression
    - RandomForestRegressor
6. Calcula métricas: RMSE, MAE, R² (en train y test).
7. Guarda las predicciones de TEST en la tabla `model_results`.
8. Guarda las métricas agregadas en una tabla `model_metrics`.

Uso típico (desde la raíz del proyecto):
    python .\Scripts\train_model.py --db_path .\data\artifacts\database.db --artifacts_dir .\data\artifacts

Requisitos:
    - numpy
    - scipy
    - scikit-learn
    - sqlite3 (módulo estándar de Python)
"""

import argparse
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------

def load_X_y(artifacts_dir: Path):
    """Carga X e y desde artifacts_dir.

    Busca:
        - X_preprocessed.npz (sparse)
        - o X_preprocessed.npy (dense)
        - y.npy
    """
    X_npz = artifacts_dir / "X_preprocessed.npz"
    X_npy = artifacts_dir / "X_preprocessed.npy"
    y_path = artifacts_dir / "y.npy"

    if not y_path.exists():
        raise FileNotFoundError(f"No se encontró y.npy en {artifacts_dir}")

    y = np.load(y_path)

    if X_npz.exists():
        X = sp.load_npz(X_npz)
        is_sparse = True
    elif X_npy.exists():
        X = np.load(X_npy)
        is_sparse = False
    else:
        raise FileNotFoundError(f"No se encontró X_preprocessed (.npz o .npy) en {artifacts_dir}")

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X e y tienen distinta cantidad de filas: X={X.shape[0]}, y={y.shape[0]}")

    return X, y, is_sparse


def maybe_sample(X, y, max_samples: int, random_state: int = 42):
    """Reduce el dataset a lo sumo max_samples filas.

    Esto ayuda a evitar problemas de memoria, especialmente con RandomForest.
    """
    n = X.shape[0]
    if n <= max_samples:
        return X, y, np.arange(n)

    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=max_samples, replace=False)
    if sp.issparse(X):
        X_sample = X[idx]
    else:
        X_sample = X[idx]
    y_sample = y[idx]
    return X_sample, y_sample, idx


def ensure_model_metrics_table(conn: sqlite3.Connection):
    """Crea la tabla model_metrics si no existe."""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            split TEXT NOT NULL,
            rmse REAL,
            mae REAL,
            r2 REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()


def insert_predictions(
    conn: sqlite3.Connection,
    run_id: str,
    model_name: str,
    split: str,
    row_indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
):
    """Inserta predicciones fila a fila en model_results.

    Se asume que la tabla `model_results` ya fue creada por load_to_sqlite.py con la siguiente forma:
        (id, run_id, row_idx, y_true, y_pred, split, model_name, created_at)
    """
    cur = conn.cursor()
    rows = []
    for idx, yt, yp in zip(row_indices, y_true, y_pred):
        rows.append((run_id, int(idx), float(yt), float(yp), split, model_name))

    cur.executemany(
        """
        INSERT INTO model_results (run_id, row_idx, y_true, y_pred, split, model_name)
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        rows,
    )
    conn.commit()


def insert_metrics(
    conn: sqlite3.Connection,
    run_id: str,
    model_name: str,
    split: str,
    rmse: float,
    mae: float,
    r2: float,
):
    """Inserta métricas agregadas en model_metrics."""
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO model_metrics (run_id, model_name, split, rmse, mae, r2)
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        (run_id, model_name, split, rmse, mae, r2),
    )
    conn.commit()


def train_and_evaluate_model(
    model,
    model_name: str,
    X_train,
    X_test,
    y_train,
    y_test,
    global_idx_train: np.ndarray,
    global_idx_test: np.ndarray,
    conn: sqlite3.Connection,
    run_id: str,
):
    """Entrena un modelo, calcula métricas y guarda resultados en la BD."""

    print(f"\n[INFO] Entrenando modelo: {model_name}")
    model.fit(X_train, y_train)

    # Predicciones
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Métricas
    rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)

    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    print(f"[{model_name}] TRAIN -> RMSE={rmse_train:.2f} / MAE={mae_train:.2f} / R2={r2_train:.4f}")
    print(f"[{model_name}] TEST  -> RMSE={rmse_test:.2f} / MAE={mae_test:.2f} / R2={r2_test:.4f}")

    # Guardar predicciones SOLO de test (para no llenar la BD)
    insert_predictions(conn, run_id, model_name, "test", global_idx_test, y_test, y_pred_test)

    # Guardar métricas (train y test)
    insert_metrics(conn, run_id, model_name, "train", rmse_train, mae_train, r2_train)
    insert_metrics(conn, run_id, model_name, "test", rmse_test, mae_test, r2_test)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos de regresión para Properati")
    parser.add_argument(
        "--db_path",
        type=str,
        default="data/artifacts/database.db",
        help="Ruta al archivo SQLite (database.db)",
    )
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default="data/artifacts",
        help="Directorio donde están X_preprocessed.* e y.npy",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proporción del conjunto de test (default 0.2)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=150000,
        help="Máximo de filas a usar para entrenar (para evitar problemas de memoria)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Semilla para la aleatoriedad",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    artifacts_dir = Path(args.artifacts_dir)

    if not db_path.exists():
        raise SystemExit(f"No se encontró la base de datos en: {db_path.resolve()}")
    if not artifacts_dir.exists():
        raise SystemExit(f"No se encontró el directorio de artifacts en: {artifacts_dir.resolve()}")

    # 1) Cargar X e y
    print(f"[INFO] Cargando X e y desde: {artifacts_dir}")
    X, y, is_sparse = load_X_y(artifacts_dir)
    n_rows = X.shape[0]
    print(f"[INFO] Dataset completo: {n_rows} filas, {X.shape[1]} columnas")

    # 2) Submuestreo opcional para no explotar memoria
    X_used, y_used, global_idx = maybe_sample(X, y, max_samples=args.max_samples, random_state=args.random_state)
    print(f"[INFO] Filas usadas para entrenamiento: {X_used.shape[0]}")

    # 3) Train/Test split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_used,
        y_used,
        np.arange(X_used.shape[0]),
        test_size=args.test_size,
        random_state=args.random_state,
    )
    # Mapear índices locales a índices globales (respecto del dataset original)
    global_idx_train = global_idx[idx_train]
    global_idx_test = global_idx[idx_test]

    # 4) Conectar a la BD y asegurar tabla de métricas
    print(f"[INFO] Conectando a SQLite: {db_path}")
    conn = sqlite3.connect(str(db_path))
    ensure_model_metrics_table(conn)

    # Identificador de corrida
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    print(f"[INFO] run_id = {run_id}")

    # 5) Definir modelos
    models = []

    # Linear Regression (acepta sparse)
    lin_reg = LinearRegression()
    models.append(("LinearRegression", lin_reg))

    # RandomForestRegressor (no siempre maneja bien sparse -> convertimos si hace falta)
    if is_sparse:
        print("[WARN] X es dispersa; RandomForest no acepta sparse directamente. Se usará toarray() internamente.")
        X_train_rf = X_train.toarray()
        X_test_rf = X_test.toarray()
    else:
        X_train_rf = X_train
        X_test_rf = X_test

    rf_reg = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        n_jobs=-1,
        random_state=args.random_state,
    )

    # 6) Entrenar y evaluar cada modelo
    # LinearRegression usa X_train / X_test tal cual
    train_and_evaluate_model(
        model=lin_reg,
        model_name="LinearRegression",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        global_idx_train=global_idx_train,
        global_idx_test=global_idx_test,
        conn=conn,
        run_id=run_id,
    )

    # RandomForestRegressor usa X_train_rf / X_test_rf (densos)
    train_and_evaluate_model(
        model=rf_reg,
        model_name="RandomForestRegressor",
        X_train=X_train_rf,
        X_test=X_test_rf,
        y_train=y_train,
        y_test=y_test,
        global_idx_train=global_idx_train,
        global_idx_test=global_idx_test,
        conn=conn,
        run_id=run_id,
    )

    conn.close()
    print("\n[OK] Entrenamiento terminado. Resultados guardados en:")
    print(f"    - Tabla model_results (predicciones de test)")
    print(f"    - Tabla model_metrics (métricas por modelo y split)")


if __name__ == "__main__":
    main()
