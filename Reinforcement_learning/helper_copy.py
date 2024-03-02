import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(mean_loss):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Loss')
    plt.plot(mean_loss)
    plt.ylim(ymin=0)
    plt.text(len(mean_loss)-1, mean_loss[-1], str(mean_loss[-1]))
    plt.show(block=False)
    plt.pause(.1)
