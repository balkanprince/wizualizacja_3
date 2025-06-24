from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Predykcje na zbiorze testowym (już zrobione w Twoim kodzie)
y_pred = grid_search.predict(X_test)

# Obliczenie metryk
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Wyświetlenie metryk
print("\nMetryki na zbiorze testowym:")
print(f"Dokładność (Accuracy): {accuracy:.4f}")
print(f"Precyzja (Precision): {precision:.4f}")
print(f"Czułość (Recall): {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Obliczenie macierzy omyłek
cm = confusion_matrix(y_test, y_pred)

# Wyświetlenie macierzy omyłek
print("\nMacierz omyłek:")
print(cm)

# Wizualizacja macierzy omyłek
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Nie-Spam (0)', 'Spam (1)'], yticklabels=['Nie-Spam (0)', 'Spam (1)'])
plt.xlabel('Przewidywana etykieta')
plt.ylabel('Prawdziwa etykieta')
plt.title('Macierz omyłek dla modelu Random Forest')
plt.show()