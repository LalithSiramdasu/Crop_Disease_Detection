import matplotlib.pyplot as plt
import pickle

# Load history object (if saved in training)
# In train_model.py, add this line at the end to save:
# with open("training_history.pkl", "wb") as f:
#     pickle.dump(history.history, f)

# Load training history
with open("training_history.pkl", "rb") as f:
    history = pickle.load(f)

# Plot Accuracy
plt.figure(figsize=(8,6))
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.show()

# Plot Loss
plt.figure(figsize=(8,6))
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()
