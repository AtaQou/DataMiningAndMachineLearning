import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Load the data from the kmeans clustering CSV file
df = pd.read_csv('data_clustered.csv')

features = [
    'Flow Duration',
    'Total Fwd Packet',
    'Total Bwd packets',
    'Fwd Packet Length Max',
    'Bwd Packet Length Max',
    'Fwd Packet Length Mean',
    'Bwd Packet Length Mean',
    'Flow IAT Mean'
]

# Ensure labels are clean
df['Label'] = df['Label'].astype(str).str.strip().str.capitalize()
df['Traffic Type'] = df['Traffic Type'].astype(str).str.strip()

# Show class counts for verification
print("Label distribution:\n", df['Label'].value_counts())
print("Traffic Type distribution:\n", df['Traffic Type'].value_counts())

scaler = StandardScaler()
X = scaler.fit_transform(df[features])

# ---------- Binary Classification: Label (Benign/Malicious) ----------
le_label = LabelEncoder()
y_label = le_label.fit_transform(df['Label'])

if len(le_label.classes_) < 2:
    print(f"\n[WARNING] Only one label value found: {le_label.classes_[0]}. Binary classification cannot proceed.")
else:
    # Stratify ensures both classes present in train/test splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_label, test_size=0.2, random_state=42, stratify=y_label
    )

    # SVM
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    print("==== SVM Results (Label) ====")
    print(classification_report(y_test, y_pred_svm, target_names=le_label.classes_))
    print(confusion_matrix(y_test, y_pred_svm))

    # Neural Net
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=X.shape[1]))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # For binary
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)
    y_pred_nn = (model.predict(X_test) > 0.5).astype(int).flatten()
    print("==== Neural Network Results (Label) ====")
    print(classification_report(y_test, y_pred_nn, target_names=le_label.classes_))
    print(confusion_matrix(y_test, y_pred_nn))

# ---------- Multiclass Classification: Traffic Type ----------
le_type = LabelEncoder()
y_type = le_type.fit_transform(df['Traffic Type'])

if len(le_type.classes_) < 2:
    print(f"\n[WARNING] Only one traffic type found: {le_type.classes_[0]}. Multi-class classification cannot proceed.")
else:
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        X, y_type, test_size=0.2, random_state=42, stratify=y_type
    )

    # SVM (multiclass)
    svm_type = SVC(kernel='rbf')
    svm_type.fit(X_train_t, y_train_t)
    y_pred_svm_type = svm_type.predict(X_test_t)
    print("==== SVM Results (Traffic Type) ====")
    print(classification_report(y_test_t, y_pred_svm_type, target_names=le_type.classes_))
    print(confusion_matrix(y_test_t, y_pred_svm_type))

    # Neural Net (multiclass)
    n_classes = len(le_type.classes_)
    model_type = Sequential()
    model_type.add(Dense(32, activation='relu', input_dim=X.shape[1]))
    model_type.add(Dense(16, activation='relu'))
    model_type.add(Dense(n_classes, activation='softmax'))
    model_type.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_type.fit(X_train_t, y_train_t, epochs=30, batch_size=16, verbose=0)
    y_pred_nn_type = np.argmax(model_type.predict(X_test_t), axis=1)
    print("==== Neural Network Results (Traffic Type) ====")
    print(classification_report(y_test_t, y_pred_nn_type, target_names=le_type.classes_))
    print(confusion_matrix(y_test_t, y_pred_nn_type))