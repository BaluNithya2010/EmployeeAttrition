import joblib

# Save a trained model to a file
def save_model(model, model_filename):
    joblib.dump(model, model_filename)
    print(f"Model saved as {model_filename}")
