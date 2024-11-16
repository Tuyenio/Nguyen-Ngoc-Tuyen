import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model

# Load dữ liệu IRIS
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hàm đánh giá mô hình
def evaluate_model(model, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, train_time, y_pred

# KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn_accuracy, knn_time, knn_pred = evaluate_model(knn, X_train, X_test, y_train, y_test)

# SVM
svm = SVC(kernel='linear')
svm_accuracy, svm_time, svm_pred = evaluate_model(svm, X_train, X_test, y_train, y_test)

# ANN
ann = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])
ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

start_time = time.time()
ann.fit(X_train, y_train, epochs=50, batch_size=4, verbose=0)
ann_time = time.time() - start_time
ann_accuracy = ann.evaluate(X_test, y_test, verbose=0)[1]
ann_pred = np.argmax(ann.predict(X_test), axis=1)

# Vẽ confusion matrix không cần seaborn
def plot_confusion_matrix_simple(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(iris.target_names))
    plt.xticks(tick_marks, iris.target_names, rotation=45)
    plt.yticks(tick_marks, iris.target_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Ghi số liệu vào các ô
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.show()

# Hiển thị kết quả
print("=== Evaluation Results ===")
results = [
    {"Model": "KNN", "Accuracy": knn_accuracy, "Time (s)": knn_time},
    {"Model": "SVM", "Accuracy": svm_accuracy, "Time (s)": svm_time},
    {"Model": "ANN", "Accuracy": ann_accuracy, "Time (s)": ann_time}
]

for result in results:
    print(f"{result['Model']}: Accuracy = {result['Accuracy']:.4f}, Time = {result['Time (s)']:.4f} seconds")

# Vẽ Confusion Matrix cho từng mô hình
plot_confusion_matrix_simple(y_test, knn_pred, "KNN Confusion Matrix")
plot_confusion_matrix_simple(y_test, svm_pred, "SVM Confusion Matrix")
plot_confusion_matrix_simple(y_test, ann_pred, "ANN Confusion Matrix")

# Lưu mô hình ANN
ann.save("iris_ann_model.h5")
print("\nANN Model saved as 'iris_ann_model.h5'")

# Vẽ biểu đồ ANN Architecture
plot_model(ann, to_file="ann_model_structure.png", show_shapes=True)
print("\nANN model structure saved as 'ann_model_structure.png'")
