import os
import pandas as pd
import numpy as np
import cv2
from sklearn . model_selection import train_test_split
from sklearn . preprocessing import LabelEncoder , OneHotEncoder
from sklearn . metrics import confusion_matrix ,classification_report
import matplotlib . pyplot as plt
import seaborn as sns

# Fonctions d’activation
def relu(x):
    """
    ReLU activation: max(0, x)
    """
    assert isinstance(x, np.ndarray), "Input to ReLU must be a numpy array"
    result = np.maximum(0, x)  
    assert np.all(result >= 0), "ReLU output must be non-negative"
    return result

def relu_derivative(x):
    """
    Derivative of ReLU: 1 if x > 0, else 0
    """
    assert isinstance(x, np.ndarray), "Input to ReLU derivative must be a numpy array"
    result = (x > 0).astype(float)  
    assert np.all((result == 0) | (result == 1)), "ReLU derivative must be 0 or 1"
    return result

def softmax(x):
    """
    Softmax activation: exp(x) / sum(exp(x))
    """
    assert isinstance(x, np.ndarray), "Input to softmax must be a numpy array"
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) 
    result = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    assert np.all((result >= 0) & (result <= 1)), "Softmax output must be in [0, 1]"
    assert np.allclose(np.sum(result, axis=1), 1), "Softmax output must sum to 1 per sample"
    return result

class MultiClassNeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        """
        Initialise le réseau de neurones à plusieurs couches.
        
        Paramètres :
        - layer_sizes : liste d'entiers [input_size, hidden1_size, ..., output_size]
        - learning_rate : taux d'apprentissage
        """
        # Vérifications de validité des paramètres
        assert isinstance(layer_sizes, list) and len(layer_sizes) >= 2, \
            "layer_sizes doit être une liste avec au moins 2 éléments"
        assert all(isinstance(size, int) and size > 0 for size in layer_sizes), \
            "Tous les éléments de layer_sizes doivent être des entiers positifs"
        assert isinstance(learning_rate, (int, float)) and learning_rate > 0, \
            "learning_rate doit être un nombre positif"

        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []

        # Initialisation des poids et biais
        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01
            b = np.zeros((1, layer_sizes[i + 1]))

            assert w.shape == (layer_sizes[i], layer_sizes[i + 1]), \
                f"Poids {i+1} a une forme incorrecte"
            assert b.shape == (1, layer_sizes[i + 1]), \
                f"Biais {i+1} a une forme incorrecte"

            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        """
        Forward propagation : Z[l] = A[l-1]W[l] + b[l], A[l] = g(Z[l])
        """
        assert isinstance(X, np.ndarray), "Input X must be a numpy array"
        assert X.shape[1] == self.layer_sizes[0], f"Input dimension ({X.shape[1]}) must match input layer size ({self.layer_sizes[0]})"
    
        self.activations = [X]
        self.z_values = []
    
        for i in range(len(self.weights) - 1):
            z = self.activations[-1] @ self.weights[i] + self.biases[i]  
            assert z.shape == (X.shape[0], self.layer_sizes[i + 1]), f"Z^{i+1} has incorrect shape"
            self.z_values.append(z)
            self.activations.append(relu(z))
    
        z = self.activations[-1] @ self.weights[-1] + self.biases[-1]  
        assert z.shape == (X.shape[0], self.layer_sizes[-1]), "Output Z has incorrect shape"
        self.z_values.append(z)
        output = softmax(z)  
        assert output.shape == (X.shape[0], self.layer_sizes[-1]), "Output A has incorrect shape"
        self.activations.append(output)
    
        return self.activations[-1]
    
    def compute_loss(self, y_true, y_pred):
        """
        Categorical Cross-Entropy
        """
        assert isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray), "Inputs to loss must be numpy arrays"
        assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
    
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))  
        assert not np.isnan(loss), "Loss computation resulted in NaN"
        return loss
    
    def compute_accuracy(self, y_true, y_pred):
        """
        Compute accuracy
        """
        assert isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray), "Inputs to accuracy must be numpy arrays"
        assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
    
        predictions = np.argmax(y_pred, axis=1) 
        true_labels = np.argmax(y_true, axis=1)  
        accuracy = np.mean(predictions == true_labels)  
        assert 0 <= accuracy <= 1, "Accuracy must be between 0 and 1"
        return accuracy
    
    def backward(self, X, y, outputs):
        """
        Backpropagation
        """
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray) and isinstance(outputs, np.ndarray), "Inputs to backward must be numpy arrays"
        assert X.shape[1] == self.layer_sizes[0], "Input dimension mismatch"
        assert y.shape == outputs.shape, "y and outputs must have same shape"
    
        m = X.shape[0]
        self.d_weights = [None] * len(self.weights)
        self.d_biases = [None] * len(self.biases)
    
        # Output layer
        dZ = outputs - y
        assert dZ.shape == outputs.shape, "dZ for output layer has incorrect shape"
        self.d_weights[-1] = (self.activations[-2].T @ dZ) / m
        self.d_biases[-1] = np.sum(dZ, axis=0, keepdims=True) / m
    
        # Hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            dA = dZ @ self.weights[i + 1].T
            dZ = dA * relu_derivative(self.z_values[i])
            assert dZ.shape == (X.shape[0], self.layer_sizes[i + 1]), f"dZ^{i+1} has incorrect shape"
            self.d_weights[i] = (self.activations[i].T @ dZ) / m
            self.d_biases[i] = np.sum(dZ, axis=0, keepdims=True) / m
    
        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.d_weights[i]
            self.biases[i] -= self.learning_rate * self.d_biases[i]
    def train(self, X, y, X_val, y_val, epochs, batch_size):
        """
        Train the neural network using mini-batch SGD, with validation.
        """
        # Vérifications des dimensions et types
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray), "X and y must be numpy arrays"
        assert isinstance(X_val, np.ndarray) and isinstance(y_val, np.ndarray), "X_val and y_val must be numpy arrays"
        assert X.shape[1] == self.layer_sizes[0], f"Input dimension ({X.shape[1]}) must match input layer size ({self.layer_sizes[0]})"
        assert y.shape[1] == self.layer_sizes[-1], f"Output dimension ({y.shape[1]}) must match output layer size ({self.layer_sizes[-1]})"
        assert X_val.shape[1] == self.layer_sizes[0], f"Validation input dimension ({X_val.shape[1]}) must match input layer size ({self.layer_sizes[0]})"
        assert y_val.shape[1] == self.layer_sizes[-1], f"Validation output dimension ({y_val.shape[1]}) must match output layer size ({self.layer_sizes[-1]})"
        assert isinstance(epochs, int) and epochs > 0, "Epochs must be a positive integer"
        assert isinstance(batch_size, int) and batch_size > 0, "Batch size must be a positive integer"
    
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
    
        for epoch in range(epochs):
            # Shuffle les données d'entraînement
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
    
            epoch_loss = 0
    
            # Mini-batch training
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
    
                outputs = self.forward(X_batch)
                loss = self.compute_loss(y_batch, outputs)
                epoch_loss += loss
    
                self.backward(X_batch, y_batch, outputs)
    
            # Moyenne de la perte sur l'époque
            train_loss = epoch_loss / (X.shape[0] // batch_size)
    
            # Calcul précision et perte sur train + val
            train_pred = self.forward(X)
            train_accuracy = self.compute_accuracy(y, train_pred)
    
            val_pred = self.forward(X_val)
            val_loss = self.compute_loss(y_val, val_pred)
            val_accuracy = self.compute_accuracy(y_val, val_pred)
    
            # Stockage des métriques
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
    
            # Affichage toutes les 10 époques
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
    
        return train_losses, val_losses, train_accuracies, val_accuracies
    def predict(self, X):
        """
        Predict class labels for the given input data.
    
        Parameters:
        - X (np.ndarray): Input data of shape (n_samples, n_features)
    
        Returns:
        - np.ndarray: Predicted class labels (n_samples,)
        """
        # Vérification des types et dimensions
        assert isinstance(X, np.ndarray), "Input X must be a numpy array"
        assert X.shape[1] == self.layer_sizes[0], (
            f"Input dimension ({X.shape[1]}) must match input layer size ({self.layer_sizes[0]})"
        )
    
        # Propagation avant pour obtenir les scores
        outputs = self.forward(X)
    
        # Prédiction : classe avec la probabilité maximale
        predictions = np.argmax(outputs, axis=1)
    
        # Vérification de la forme des prédictions
        assert predictions.shape == (X.shape[0],), "Predictions have incorrect shape"
    
        return predictions

# Définir le chemin vers le dossier décompressé
data_dir = os.path.join(os.getcwd(), 'amhcd-data-64/tifinagh-images/')
print("Chemin vers les données :", data_dir)

current_working_directory = os.getcwd()
print("Répertoire courant :", current_working_directory)

# Charger le fichier CSV contenant les étiquettes
try:
    csv_path = os.path.join(data_dir, 'labels-map.csv')
    labels_df = pd.read_csv(csv_path)

    assert 'image_path' in labels_df.columns and 'label' in labels_df.columns, \
        "Le CSV doit contenir les colonnes 'image_path' et 'label'"

except FileNotFoundError:
    print("labels-map.csv introuvable. Construction du DataFrame à partir des dossiers...")

    image_paths = []
    labels = []

    for label_dir in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_dir)

        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                image_paths.append(os.path.join(label_path, img_name))
                labels.append(label_dir)

    labels_df = pd.DataFrame({
        'image_path': image_paths,
        'label': labels
    })

# Afficher un aperçu
print("Nombre total d’images :", len(labels_df))
print(labels_df.head())





# Charger et prétraiter toutes les images
X = np.array([
    load_and_preprocess_image(os.path.join(data_dir, path))
    for path in labels_df['image_path']
])
y = labels_df['label_encoded'].values

# Vérifications
assert X.shape[0] == y.shape[0], "Mismatch entre le nombre d’images et de labels"
assert X.shape[1] == 32 * 32, f"Les images doivent être aplaties à {32*32} pixels, obtenu : {X.shape[1]}"

# Split des données (80% train+val, 20% test puis 60% train / 20% val)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

# Convertir explicitement en NumPy arrays
X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

# Vérification finale de la cohérence des tailles
assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == X.shape[0], "Les tailles train/val/test ne correspondent pas au total"

print(f"Train : {X_train.shape[0]} échantillons, "
      f"Validation : {X_val.shape[0]} échantillons, "
      f"Test : {X_test.shape[0]} échantillons")

# Encodage one-hot
one_hot_encoder = OneHotEncoder(sparse_output=False)
y_train_one_hot = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
y_val_one_hot = one_hot_encoder.transform(y_val.reshape(-1, 1))
y_test_one_hot = one_hot_encoder.transform(y_test.reshape(-1, 1))

# Vérification des types
assert isinstance(y_train_one_hot, np.ndarray), "y_train_one_hot doit être un np.array"
assert isinstance(y_val_one_hot, np.ndarray), "y_val_one_hot doit être un np.array"
assert isinstance(y_test_one_hot, np.ndarray), "y_test_one_hot doit être un np.array"

# Initialisation et entraînement du modèle
layer_sizes = [X_train.shape[1], 64, 32, num_classes]  #  1024 -> 64 -> 32 -> 33
nn = MultiClassNeuralNetwork(layer_sizes, learning_rate=0.01)

train_losses, val_losses, train_accuracies, val_accuracies = nn.train(
    X_train, y_train_one_hot,
    X_val, y_val_one_hot,
    epochs=100,
    batch_size=32
)

# Prédictions
y_pred = nn.predict(X_test)

print("\nRapport de classification (jeu de test) :")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Matrice de confusion (Test set)')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.savefig('confusion_matrix.png')
plt.show()

# Courbes de perte et d’accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Perte
ax1.plot(train_losses, label='Train Loss')
ax1.plot(val_losses, label='Validation Loss')
ax1.set_title('Courbe de perte')
ax1.set_xlabel('Époque')
ax1.set_ylabel('Perte')
ax1.legend()


# Précision
ax2.plot(train_accuracies, label='Train Accuracy')
ax2.plot(val_accuracies, label='Validation Accuracy')
ax2.set_title('Courbe de précision')
ax2.set_xlabel('Époque')
ax2.set_ylabel('Précision')
ax2.legend()




plt.tight_layout()
fig.savefig('loss_accuracy_plot.png')
plt.show()
