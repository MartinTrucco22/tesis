import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import recall_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter

from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD

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

# Lista para almacenar historias de entrenamiento
history_list = []


for i in range(1, long_camadas):

    X = result.loc[result.index.get_level_values('camada') == i]
    y = result['Max_Sit_deuda'].loc[result.index.get_level_values('camada') == i+1]

    smote = SMOTE(sampling_strategy=0.4, random_state=42)

    X_resampled, y_resampled = smote.fit_resample(X.values, y.values[:,-1])
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    proporcion_original_y = Counter(y.iloc[:,-1])
    proporcion_equilibrado_y = Counter(y_resampled)
    print("Proporción de clases en el conjunto de datos   original :", proporcion_original_y)
    print("Proporción de clases en el conjunto de datos equilibrado:", proporcion_equilibrado_y)

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    
    model = Sequential()
    model.add(LSTM(units=50, activation='sigmoid', return_sequences=True, input_shape=(1, X_train.shape[1])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    
    if modelo_base is not None:
        model.set_weights(modelo_base.get_weights())
    
    learning_rate = 0.1
    optimizer = SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])

    # callbacks para registrar la pérdida durante el entrenamiento
    history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), callbacks=[EarlyStopping(patience=5, monitor='val_loss', mode='min')])
    history_list.append(history)
    np.save(f'historial_{i}.npy', history.history)

    y_prob = model.predict(X_test)
    
    # Establecemos el modelo entrenado como el modelo base para la próxima iteración
    modelo_base = model


    ##########################################
    plt.plot(history.history['loss'], label='Error de entrenamiento')
    plt.plot(history.history['val_loss'], label='Error de testeo')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()

    plt.savefig(f'errores_{i}.png')
    plt.close()
    ##########################################

# Guardamos el último modelo entrenado después de todas las iteraciones
joblib.dump(modelo_base, 'LSTM_Final.joblib')


# Métricas

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
optimal_threshold_index = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_threshold_index]
y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
recall_final = recall_score(y_test, y_pred_optimal)
cm = confusion_matrix(y_test, y_pred_optimal)

np.save('LSTM_fpr.npy', fpr)
np.save('LSTM_tpr.npy', tpr)
np.save('LSTM_cm.npy', cm)
np.save('LSTM_y_train.npy', y_train)
np.save('LSTM_y_test.npy', y_test)
np.save('LSTM_y_prob.npy', y_prob)
np.save('LSTM_y_pred_optimal.npy', y_pred_optimal)

file1 = open('LSTM_Metricas.txt', 'w')
file1.write(f'-------------------------------- Métricas de la red LSTM ---------------------------------\n')
file1.write('Pérdida en entrenamiento: '+ str(history_list[-1].history['loss'][-1]) + '\n')
file1.write('Pérdida en validación: ' + str(history_list[-1].history['val_loss'][-1]) + '\n')
file1.write('Área bajo la curva: ' + str(auc) + '\n')
file1.write('Umbral óptimo: ' + str(optimal_threshold) + '\n')
file1.write(f'Recall del último modelo para el umbral óptimo: {recall_final:.5f}\n')
file1.write('FPR_scatter del umbral: ' + str(fpr[optimal_threshold_index]) + '\n')
file1.write('TPR_scatter del umbral: ' + str(tpr[optimal_threshold_index]) + '\n')
file1.write('argmax(TPR - FPR): ' + str(optimal_threshold_index) + '\n')
file1.write(classification_report(y_test, y_pred_optimal, digits=6))
file1.close()
