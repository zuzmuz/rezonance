import torch
import matplotlib.pyplot as plt

from rezonance import transforms


def main():
    pitches = torch.arange(64, 69)
    classification = transforms.NoteClassifier(60, 72, 0.5)

    classes = classification.forward(pitches)

    plt.imshow(classes)
    plt.show()


if __name__ == "__main__":
    main()
