import matplotlib.pyplot as plt
import numpy as np

def main():
    # Some example data to display
    x = np.linspace(0, 2 * np.pi, 400)
    y = np.sin(x ** 2) 

    fig, ax = plt.subplots()
    ax.set_title('A single plot')
    ax.plot(x, y)

    return fig 
