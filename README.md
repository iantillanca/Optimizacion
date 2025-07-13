# Proyecto: Clasificación de Costos Médicos Personales

## Descripción  
En este proyecto aplicamos un pipeline completo de machine learning para clasificar rangos de costo médico en categorías (`low`, `med`, `high`) a partir del **Medical Cost Personal Dataset**. El objetivo es demostrar un flujo de trabajo end-to-end: exploración y limpieza de datos, diseño de variables objetivo, preprocesamiento, entrenamiento de modelos, optimización de hiperparámetros y evaluación.

## Dataset  
- **Origen:** Kaggle – “Medical Cost Personal Dataset”  
- **Tamaño:** ~1.300 filas, 7 columnas originales  
- **Variables clave:**  
  - `age`, `sex`, `bmi`, `children`, `smoker`, `region`  
  - `charges` (costo continuo → discretizado en 3 clases)

## Estructura del Repositorio  
/Proyecto_Costos_Medicos
├─ insurance.csv # Datos originales
├─ notebook.ipynb # Jupyter Notebook con todo el flujo
└─ README.md # Este documento


## Resumen del EDA Inicial  
1. **Valores faltantes y duplicados:** No se detectaron.  
2. **Distribución de `charges`:** Sesgada a la derecha, unos pocos casos con costos muy altos.  
3. **Correlaciones numéricas:**  
   - `bmi` y `charges` muy correlacionados.  
   - `smoker` vs `charges`: salto claro en pacientes fumadores.  
4. **Outliers:** Valores extremos de `charges` detectados y discretizados en terciles.

## Definición del Problema  
- **Tipo:** Clasificación multiclase  
- **Target:** `cost_cat` en 3 niveles (`low`, `med`, `high`) generados por percentiles de `charges`.  
- **Justificación:** Convertir un problema de regresión en clasificación permite predecir rangos de costo y facilita la toma de decisión en seguros médicas.

## Preprocesamiento  
- Imputación de valores numéricos con mediana.  
- One-Hot Encoding de `sex`, `smoker` y `region`.  
- Escalado de variables numéricas con `StandardScaler`.  
- Pipeline único con `ColumnTransformer`.

## Modelos y Optimización  
1. **Modelos comparados:**  
   - Regresión Logística  
   - K-Nearest Neighbors (KNN)  
   - Random Forest (RF)  
2. **Validación inicial:** accuracy, precision, recall, F1-score  
3. **Optimización:**  
   - **KNN:** GridSearchCV sobre `n_neighbors` ∈ [3…15], `weights` ∈ {`uniform`,`distance`}  
   - **RF:** GridSearchCV sobre `n_estimators` ∈ [50,100], `max_depth` ∈ {None,10,20}  
   - Se trabajó también con `class_weight='balanced'` en RF para corregir el desequilibrio.

## Resultados Principales  
| Modelo                | Accuracy | Precision | Recall | F1-score (weighted) |
|-----------------------|---------:|----------:|-------:|--------------------:|
| Logistic Regression   | 0.60     | 0.58      | 0.60   | 0.58                |
| KNN (optimizado)      | 0.68     | 0.65      | 0.68   | 0.66                |
| Random Forest (opt.)  | 0.68     | 0.67      | 0.68   | 0.67                |
| RF (balanced weights) | 0.69     | 0.67      | 0.69   | 0.67                |

> **Conclusión:**  
> - **KNN** y **Random Forest** empatan en desempeño (F1≈0.67).  
> - RF con `class_weight='balanced'` mejora ligeramente la recall de clases minoritarias.  
> - Para producción, **Random Forest** balanceado ofrece mayor robustez y facilidad de interpretación de importancia de variables.

## Próximos Pasos  
- Explorar técnicas de ensamble (e.g. XGBoost, LightGBM).  
- Refinar discretización del target (¿otros cut-points?).  
- Validar con nuevas muestras externas.

## Cómo ejecutar en Colab  
1. Subir `insurance.csv` a tu Google Drive (`MyDrive/Datasets/insurance.csv`).  
2. Abrir el notebook en Colab.  
3. Montar tu Drive al inicio del notebook:  
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   df = pd.read_csv('/content/drive/MyDrive/Datasets/insurance.csv')
