from keras.models import load_model

model_file = "./results/final_datasets_rnn/models_2024-04-03T155223/rnn_100pct_2L_70.keras"

model = load_model(model_file)

model.save("./results/final_datasets_rnn/models_2024-04-03T155223/rnn_100pct_2L_70.h5", save_format='h5')
