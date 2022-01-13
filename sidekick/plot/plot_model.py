from tensorflow.keras.utils import plot_model

def visualize_model(model, filename):
    plot_model(model, to_file=filename, show_shapes=True)