import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
"""
* Rozpoznawanie zwierząt na podstawie datasetu CIFAR10 - CNN*

https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html

Autorzy:
- Bartosz Krystowski s19545
- Robert Brzoskowski s21162

Przygotowanie środowiska:
Instalacja bibliotek: torch, torchvision

"""
"""Jeśli dostępny jest procesor graficzny, użyj go do obliczeń, w przeciwnym razie użyj procesora."""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"Hyper-parameters"
num_epochs = 4
batch_size = 4
learning_rate = 0.001

"""Transformacja obrazków z PILImage o zakresie [0, 1] będącego w zestawie danych,
 na Tensory o znormalizowanym zakresie [-1, 1]"""
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

"""
Załadowanie danych testowych i treningowych przy użyciu modułu torchvision
Zbiór CIFAR10 zawiera 60000 obrazków w rozdzielczości 32x32 podzielonych
na 10 różnych klas (każda klasa to 6tys. obrazków).
"""
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=False)

"""Klasy z możliwymi nazwami zwierząt w nich"""
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

"""Utworzenie klasy ConvNet,dziedziczy klasę z nn.Module, 
będącą superklasą dla wszystkich sieci nauronowych w Pytorch"""
class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        """Conv2d  (3-in_chanels-obrazy wchodzące mają 3 kanały kolorów, 
                    6-Liczba kanałów utworzonych przez splot (convolution)
                    5-kernel size - siatka 5x5"""
        self.conv1 = nn.Conv2d(3, 6, 5)
        """MaxPool2d - Max Pooling - maksymalna siatka 
                    po każdej operacji kernela na obrazie - w tym przypadku 2x2"""
        self.pool = nn.MaxPool2d(2, 2)
        """Conv2d"""
        self.conv2 = nn.Conv2d(6, 16, 5)
        """Linear - Stosuje transformację liniową do przychodzących danych
                    (16*5*5 - rozmiar wejscia - input size
                    120 - rozmiar wyjście - output size)"""
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

"""Utworzenie obiektu / modelu z klasy ConvNet"""
model = ConvNet().to(device)
"""Definiowanie funkcji straty. używamy tutaj CrossEntropyLoss() 
- To kryterium oblicza krzyżową utratę entropii
 między logami wejściowymi a celem."""
criterion = nn.CrossEntropyLoss()
"""Optimizer, SGD - Implementuje stochastyczny spadek gradientu (opcjonalnie z pędem)."""
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
"""Pętla treningowa, zapętlamy przy pomocy num_epochs """
for epoch in range(num_epochs):
    """Pętla w pętli, zapętlamy przy pomocy train_loader """
    for i, (images, labels) in enumerate(train_loader):
        """Oryginalny obraz [4, 3, 32, 32] = 4, 3, 1024px"""
        """input_layer: 3 input channels, 6 output channels, 5 kernel size"""
        """Wypushowanie obrazów do GPU"""
        images = images.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 2500 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
"""Powiadomienie o zakońćzonym treningu"""
print('Finished Training')

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
    """Dokładność tego CNN"""
    acc = 100.00 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')
    """Dokładność dla każdej klasy osobno"""
    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')
