import numpy as np
import scipy as sc
import pandas as pd
import sklearn as sk
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn import svm
from scipy.stats import spearmanr
from tensorflow import keras
from keras.utils import to_categorical

#carga de datos
adults = pd.read_csv('explainableAI/datasets/adult/adult.data', header=None,
                       names=['age', 'workclass', 'fnlwgt', 'education',
                              'education-num', 'marital-status', 'occupation', 'relationship',
                                'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'money'])

attributes_adults = adults.loc[:, 'age':'native-country']
goal_adult = adults['money']

codificator_attributes = sk.preprocessing.OrdinalEncoder()
codificator_attributes.fit(attributes_adults)
codified_attributes = codificator_attributes.transform(attributes_adults)

codificator_goal = sk.preprocessing.LabelEncoder()
codified_goal = codificator_goal.fit_transform(goal_adult)

max_attributes_adults = [max(codified_attributes[x][:]) for x in range(14)]
min_attributes_adults = [min(codified_attributes[x][:]) for x in range(14)]

(training_attributes, test_attributes,
 training_goal, test_goal) = model_selection.train_test_split(
     codified_attributes,codified_goal, random_state=12345, test_size=.33, stratify=codified_goal)


#Entrenamiento de modelo random forest model para datos adult
randomForestModel = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
randomForestModel.fit(training_attributes, training_goal)

#Entrenamiento de modelo xgboost para datos adult

training_data = xgb.DMatrix(training_attributes, label= training_goal)

#Se definen los parametros para una tarea de clasificación binaria
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
}

xgboostModel = xgb.train(params, training_data)

#función para calcular la distancia coseno de dos muestras
def cosine_distance(muestra1, muestra2):

    vector1 = np.array(muestra1)
    vector2 = np.array(muestra2)

    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    cosine_distance = dot_product / (norm1 * norm2)

    return cosine_distance


def LIMEAlgorithm(data, f, N, max_attributes, min_attributes):

    #Inicalizamos las listas donde se guardaran los diferentes conjuntos
    X = []
    R = []
    W = []
    Y = []

    #Realizamos N iteraciones para generar N permutaciones de los atributos de la muestra
    for i in range (N):

    #Generamos una lista aleatoria que tenga como maximo el tamaño de la muestra y cuyos enteros esten entre 0 y el tamaño de la muestra
        attributes = np.random.choice(len(data),np.random.randint(len(data)), replace=False)
    #Generamos las sublistas que representaran los diferentes valores para 1 permutación de la muestra
        element_R = []
        element_X = []
    #Recorremos los atributos de la muestra
        for j in range(len(data)):
    #Comprobamos si el indice del atributo está en la lista generada aleatoriamente para decidir si perturbamos el atributo o no
            if j in attributes:

                element_R.append(0)
                modifiedAttribute = np.random.randint(min_attributes[j], max_attributes[j])
                element_X.append(modifiedAttribute)

            else:

                element_R.append(1)
                element_X.append(data[j])
            

        W.append(cosine_distance(element_X, data))

        X.append(element_X)
        R.append(element_R)
    #Una vez realizadas las permutaciones las ponderamos con los pesos generados en W
    R_ponderada = []

    for i in range(len(R)):

        num = W[i]
        sublist = R[i]
        R_muestra_ponderada = []

        for j in range(len(sublist)):

            res = num * sublist[j]
            R_muestra_ponderada.append(res)

        R_ponderada.append(R_muestra_ponderada)

    #Realizamos las predicciones de las permutaciones con el modelo pasado como parametro
    for i in range(len(X)):

        prediction = f.predict(np.array(X[i]).reshape(1, -1))
        Y.append(prediction)

    #Se entrena el modelo ridge pasandole las R ponderadas como atributos y las predicciones de Y como objetivo
    G = Ridge(alpha=1.0)
    G.fit(R_ponderada, Y)

    return G.get_params()

#Metricas

def identidad(muestra_1, muestra_2, explicación_1, explicación_2):

    distancia_muestras = cosine_distance(muestra_1, muestra_2)

    if distancia_muestras == 0:

        distancia_explicación = cosine_distance(explicación_1, explicación_2)

        if distancia_explicación == 0:

            print('estas muestras cumplen la metrica de identidad')

        else:

            print('estas muestras no cumplen con la metrica de identidad')

    
    else:
        
        print('estas muestras no son identicas')
        
        
        
def separabilidad(muestra_1, muestra_2,explicacion_a, explicacion_b, umbral_distancia=1e-6):
    distacia_muestras_ab=cosine_distance(muestra_1, muestra_2)
    if distancia_muestras_ab == 0:
        return True
    
    distancia_explicaciones_ab=cosine_distance(explicacion_a, explicacion_b)
    if distancia_explicaciones_ab<=umbral_distancia:
        return False
    return True

def estabilidad(explanations, perturbation):
    num_objects = len(explanations)
    distances_original = []
    distances_perturbed = []
    
    for i in range(num_objects):
        distance_original = d(explanations[i], explanations[0]) 
        distances_original.append(distance_original)
        
        distance_perturbed = d(perturbations[i], perturbations[0]) 
        distances_perturbed.append(distance_perturbed)
        
        
    correlation, _ = spearmanr(distances_original, distances_perturbed)
    
    return correlation
       
def selectividad(training_atribute, training_goal, test_atribute, test_goal):

    y_pred_orig = model.predict(test_atribute)


    auc_orig = roc_auc_score(test_goal, y_pred_orig)


    num_features = training_atribute.shape[1]


    auc_values = []


    for i in range(num_features):
        X_test_modified = test_atribute.copy()
        X_test_modified[:, i] = 0 
    
     
        y_pred_modified = model.predict(X_test_modified)
    
    
        auc_modified = roc_auc_score(training_goal, y_pred_modified)
    
  
        auc_change = auc_orig - auc_modified
    
    
        auc_values.append(auc_change)


    selectivity = np.sum(auc_values)

    return selectivity

        
def coherencia(p,e):
    diferencia=abs(p-e)
    return diferencia


def completitud(error_explicacion, error_prediccion):
    return (error_explicacion / error_prediccion) *100


def congruencia(coherencia):
    n=len(coherencia)
    promedio=sum(coherencia)/n
    suma=sum((a-promedio)**2 for a in coherencia)
    result = math.sqrt(suma/n)
    return result
    
    
d = LIMEAlgorithm(codified_attributes[:][5], randomForestModel, 8, max_attributes_adults, min_attributes_adults)