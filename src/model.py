import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    CNN para CIFAR-10.
    Entrada: tensor 4D (Batch_Size, 3, 32, 32)
    Salida: tensor (Batch_Size, 10)
    """

    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            # Bloque convolucional 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloque convolucional 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Validación obligatoria de shapes con tensor 4D simulado
    model = CNN()
    x = torch.randn(4, 3, 32, 32)

    print("Input:", tuple(x.shape))

    h = x
    for i, layer in enumerate(model.features):
        h = layer(h)
        print(f"features[{i}] - {layer.__class__.__name__}: {tuple(h.shape)}")

    h = model.classifier[0](h)  # Flatten
    print("Flatten:", tuple(h.shape))

    h = model.classifier[1](h)  # Linear 4096 -> 128
    print("Linear 1:", tuple(h.shape))

    h = model.classifier[2](h)  # ReLU
    print("ReLU:", tuple(h.shape))

    h = model.classifier[3](h)  # Linear 128 -> 10
    print("Linear 2 / Output:", tuple(h.shape))

    output = model(x)
    assert output.shape == (4, 10), "Shape mismatch: se esperaba (4, 10)"
    print("Forward pass OK. Output final:", tuple(output.shape))