import pickle
from predictor_model import load_model

model = load_model()

# Save the model as a pickle file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
