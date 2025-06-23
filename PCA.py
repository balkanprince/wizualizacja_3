import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from math import sqrt
from prettytable import PrettyTable

# Wczytaj dane
try:
    bike_data = pd.read_csv('C:\\Users\\mpiesio\\Desktop\\KODILLA\\wizualizacja\\daily-bike-share.csv')
except FileNotFoundError:
    print("Błąd: Plik daily-bike-share.csv nie istnieje. Podaj poprawną ścieżkę.")
    exit()

# Definiuj zmienne
target = 'rentals'
numeric_features = ['temp', 'atemp', 'hum', 'windspeed']  # Więcej cech numerycznych
categorical_features = ['season', 'weathersit', 'workingday']  # Cechy kategoryczne

# Eksploracyjna analiza danych (EDA)
print("=== Eksploracyjna analiza danych (EDA) ===")
print("\nPierwsze 5 wierszy danych:")
print(bike_data.head())
print("\nPodstawowe statystyki cech numerycznych:")
print(bike_data[numeric_features].describe())
print("\nRozkład cech kategorycznych:")
for col in categorical_features:
    print(f"\n{col}:\n{bike_data[col].value_counts()}")

# Wizualizacja: Korelacja cech numerycznych
plt.figure(figsize=(8, 6))
sns.heatmap(bike_data[numeric_features].corr(), annot=True, cmap='RdYlGn', vmin=-1, vmax=1)
plt.title("Korelacja Pearsona dla cech numerycznych")
plt.show()

# Pierwszy model: Prosta regresja liniowa z 'temp'
X = bike_data[['temp']].copy()
y = bike_data[target].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_first_train = lr_model.predict(X_train)
y_pred_first_test = lr_model.predict(X_test)

# Końcowy model: ElasticNet z PCA
X = bike_data[numeric_features + categorical_features].copy()
y = bike_data[target].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Definiuj transformatory
numeric_transformer = Pipeline(steps=[
    ('logtransformer', PowerTransformer()),
    ('standardscaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Pipeline z PCA i ElasticNet
pca_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('pca', PCA(random_state=42)),
    ('regressor', ElasticNet())
])

# Hiperparametry dla GridSearchCV
params = {
    'pca__n_components': [2, 3, 5, 7],  # Różna liczba głównych składowych
    'regressor__alpha': [1e-3, 1e-2, 1e-1, 1.0],
    'regressor__l1_ratio': [0.1, 0.5, 0.9]
}

# Walidacja krzyżowa
cv = KFold(n_splits=5, shuffle=False)

# GridSearchCV
pca_gridsearch = GridSearchCV(
    pca_pipeline,
    params,
    scoring='neg_mean_squared_error',
    cv=cv
)

# Trenuj model z PCA
pca_gridsearch.fit(X_train, y_train)
print("\nNajlepsze hiperparametry z PCA:", pca_gridsearch.best_params_)

# Pobierz najlepszy model
pca_model = pca_gridsearch.best_estimator_

# Predykcje z PCA
y_pred_pca_train = pca_model.predict(X_train)
y_pred_pca_test = pca_model.predict(X_test)

# Pipeline bez PCA (oryginalny ElasticNet)
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', ElasticNet())
])

# GridSearchCV bez PCA
final_gridsearch = GridSearchCV(
    final_pipeline,
    {
        'regressor__alpha': [1e-3, 1e-2, 1e-1, 1.0],
        'regressor__l1_ratio': [0.1, 0.5, 0.9]
    },
    scoring='neg_mean_squared_error',
    cv=cv
)

# Trenuj model bez PCA
final_gridsearch.fit(X_train, y_train)
print("\nNajlepsze hiperparametry bez PCA:", final_gridsearch.best_params_)

# Pobierz najlepszy model
final_model = final_gridsearch.best_estimator_

# Predykcje bez PCA
y_pred_final_train = final_model.predict(X_train)
y_pred_final_test = final_model.predict(X_test)

# Obliczenie metryk
def calculate_metrics(y_true, y_pred):
    return {
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': sqrt(mean_squared_error(y_true, y_pred))
    }

# Metryki dla wszystkich modeli
metrics_first_train = calculate_metrics(y_train, y_pred_first_train)
metrics_first_test = calculate_metrics(y_test, y_pred_first_test)
metrics_final_train = calculate_metrics(y_train, y_pred_final_train)
metrics_final_test = calculate_metrics(y_test, y_pred_final_test)
metrics_pca_train = calculate_metrics(y_train, y_pred_pca_train)
metrics_pca_test = calculate_metrics(y_test, y_pred_pca_test)

# Wyświetlenie metryk
results = PrettyTable(['Model', 'Zbiór', 'R²', 'MAE', 'MAPE', 'MSE', 'RMSE'])
def add_metrics_to_table(model_name, metrics_train, metrics_test, zbir_train='Treningowy', zbir_test='Testowy'):
    results.add_row([model_name, zbir_train, f"{metrics_train['r2']:.4f}", f"{metrics_train['mae']:.2f}", 
                     f"{metrics_train['mape']:.4f}", f"{metrics_train['mse']:.2f}", f"{metrics_train['rmse']:.2f}"])
    results.add_row([model_name, zbir_test, f"{metrics_test['r2']:.4f}", f"{metrics_test['mae']:.2f}", 
                     f"{metrics_test['mape']:.4f}", f"{metrics_test['mse']:.2f}", f"{metrics_test['rmse']:.2f}"])

add_metrics_to_table('Prosta regresja (temp)', metrics_first_train, metrics_first_test)
add_metrics_to_table('ElasticNet bez PCA', metrics_final_train, metrics_final_test)
add_metrics_to_table('ElasticNet z PCA', metrics_pca_train, metrics_pca_test)

print("\n=== Wyniki modeli ===")
print(results)

# Wizualizacja: Wyjaśniona wariancja PCA
pca = pca_model.named_steps['pca']
plt.figure(figsize=(8, 4))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, color='black')
plt.title('Wyjaśniona wariancja przez główne składowe')
plt.xlabel('Główne Składowe')
plt.ylabel('% wyjaśnionej wariancji')
plt.show()

# Wizualizacje dla modelu z PCA
# 1. Wykres punktowy: przewidywane vs. rzeczywiste
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_pca_test, color='blue', alpha=0.5, label='Predykcje')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideał')
plt.xlabel('Wartości rzeczywiste')
plt.ylabel('Wartości przewidywane')
plt.title('ElasticNet z PCA: Przewidywane vs. Rzeczywiste (Test)')
plt.legend()
plt.show()

# 2. Wykres reszt
errors_pca = y_pred_pca_test - y_test
plt.figure(figsize=(8, 6))
plt.scatter(y_test, errors_pca, color='blue', alpha=0.25)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Wartości rzeczywiste')
plt.ylabel('Reszty')
plt.title('ElasticNet z PCA: Wykres reszt (Test)')
plt.show()

# 3. Histogram reszt
plt.figure(figsize=(8, 6))
plt.hist(errors_pca, bins=20, color='blue', alpha=0.7)
plt.axvline(errors_pca.mean(), color='black', linestyle='dashed', linewidth=1)
plt.xlabel('Reszty')
plt.ylabel('Częstość')
plt.title(f'ElasticNet z PCA: Histogram reszt (Średnia = {np.round(errors_pca.mean(), 2)})')
plt.show()