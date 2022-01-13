import matplotlib.pyplot as plt
import numpy as np

def plot_graph(epochs, H, save=False):
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, epochs, 1), H.history['loss'], label='train_loss')
    plt.plot(np.arange(0, epochs, 1), H.history['val_loss'], label='val_loss')
    plt.plot(np.arange(0, epochs, 1), H.history['accuracy'], label='train_acc')
    plt.plot(np.arange(0, epochs, 1), H.history['val_accuracy'], label='val_acc')
    plt.title('Training Loss & Accuracy')
    plt.xlabel('# Epochs')
    plt.ylabel('Metric Values')
    plt.legend()
    if save==True:
        plt.savefig(fname= "./train_plot.jpg")
    plt.show()