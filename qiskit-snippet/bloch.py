from qiskit.visualization import plot_bloch_vector
import matplotlib.pyplot as plt


def main():
    bloch_vector = [0, 1, 0]

    plot_bloch_vector(bloch_vector)

    plt.show()


if __name__ == "__main__":
    main()
