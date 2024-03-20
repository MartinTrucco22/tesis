import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, log_loss, roc_curve, roc_auc_score
from sklearn.metrics import recall_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter

import warnings
warnings.filterwarnings('ignore')


df = pd.read_stata('Base_50k_cuits.dta')
df = df.drop('index', axis=1)

lista_columnas = ['Cant_Banco', 'Cant_Banco_pub', 'Max_Sit_deuda', 'Sum_Prest', 'Max_Fid', 'Max_Comp_fin',
                  'Max_Tar_Mut', 'cant_rech_1', 'monto_tot_1', 'cant_rech_2', 'monto_tot_2']

result = df.pivot_table(
    index=['camada', 'cuit'],
    columns=['t'],
    values=lista_columnas,
    aggfunc='first',
    fill_value=0
)

result['Max_Sit_deuda'] = (result['Max_Sit_deuda'] != 1).astype(int)
long_camadas = result.index.get_level_values('camada').nunique()

# Inicializamos un modelo None para la primera iteración. En las siguientes voy usando el modelo preentrenado anteriormente
modelo_base = None

for i in range(1, long_camadas):

    X = result.loc[result.index.get_level_values('camada') == i]
    y = result['Max_Sit_deuda'].loc[result.index.get_level_values('camada') == i+1]
    
    # Aplicamos SMOTE para balancear el conjunto de datos
    smote = SMOTE(sampling_strategy=0.4, random_state=42)

    X_resampled, y_resampled = smote.fit_resample(X.values, y.values[:,-1])
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    proporcion_original_y = Counter(y.iloc[:,-1])
    proporcion_equilibrado_y = Counter(y_resampled)
    print("Proporción de clases en el conjunto de datos   original :", proporcion_original_y)
    print("Proporción de clases en el conjunto de datos equilibrado:", proporcion_equilibrado_y)


    param_dist = {
    'max_iter': [10000, 100000],
    'solver': ['liblinear', 'newton-cg'],                                                       # 'lbfgs' hace renegar para converger 
    'penalty': ['l2'],
    'C': [100, 10, 1.0, 0.1, 0.01]
    }

    if modelo_base is None:
        modelo = LogisticRegression()
    else:
        modelo = modelo_base
    
    grid_search = GridSearchCV(modelo, param_grid=param_dist, cv=5, scoring='recall', random_state=42)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f'\nCamada_{i} Mejores hiperparámetros: {best_params}\n')

    best_LogReg = grid_search.best_estimator_
    best_LogReg.fit(X_train, y_train)

    y_prob = best_LogReg.predict_proba(X_test)[:,1]
    
    # Establecemos el modelo entrenado como el modelo base para la próxima iteración
    modelo_base = best_LogReg

# Guardamos el último modelo entrenado después de todas las iteraciones
joblib.dump(modelo_base, 'LogReg_Final.joblib')


# Métricas y demás

mse = mean_squared_error(y_test, y_prob)
# bce = log_loss(y_test, y_prob)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
optimal_threshold_index = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_threshold_index]
y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
recall_final = recall_score(y_test, y_pred_optimal)
cm = confusion_matrix(y_test, y_pred_optimal)

np.save('LogReg_fpr.npy', fpr)
np.save('LogReg_tpr.npy', tpr)
np.save('LogReg_cm.npy', cm)
np.save('LogReg_y_train.npy', y_train)
np.save('LogReg_y_test.npy', y_test)
np.save('LogReg_y_prob.npy', y_prob)
np.save('LogReg_y_pred_optimal.npy', y_pred_optimal)

file1 = open('LogReg_Metricas.txt', 'w')
file1.write(f'-------------------------------------- Métricas del Logistic Regression --------------------------------------\n')
file1.write('Los mejores hiperparámetros: ' + str(best_params) + '\n')
file1.write('Error cuadrático medio: '+ str(mse) + '\n')
file1.write('Área bajo la curva: ' + str(auc) + '\n')
file1.write('Umbral óptimo: ' + str(optimal_threshold) + '\n')
file1.write(f'Recall del último modelo para el umbral óptimo: {recall_final:.6f}\n')
file1.write('FPR_scatter del umbral: ' + str(fpr[optimal_threshold_index]) + '\n')
file1.write('TPR_scatter del umbral: ' + str(tpr[optimal_threshold_index]) + '\n')
file1.write('argmax(TPR - FPR): ' + str(optimal_threshold_index) + '\n')
file1.write(classification_report(y_test, y_pred_optimal, digits=6))
file1.close()