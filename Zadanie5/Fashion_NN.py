import warnings
from itertools import chain

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

"""
* Rozpoznawanie klasy (nazwy) ubrania na zasadzie datasetu FashionMNIST - NeuralNetwork*

https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html

Autorzy:
- Bartosz Krystowski s19545
- Robert Brzoskowski s21162

Przygotowanie środowiska:
Instalacja bibliotek: warnings, itertools, numpy, pandas, torch, sklearn, torchvision

"""

warnings.filterwarnings('ignore')

"""Jeśli dostępny jest procesor graficzny, użyj go do obliczeń, w przeciwnym razie użyj procesora."""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""Załadowanie danych testowych i treningowych przy użyciu modułu torchvision."""
train_set = torchvision.datasets.FashionMNIST("./data", download=True,
                                              transform=transforms.Compose([transforms.ToTensor()]))
test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False,
                                             transform=transforms.Compose([transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100)

"""
Metoda zwracająca nazwę odpowiedniej klasy dla odpowiedniego numeru
W tym datasecie mamy 10 rodzajów ubrań.
"""
def output_label(label):
    output_mapping = {
        0: "T-shirt/Top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot"
    }
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]

"""Wyświetlenie 10 obrazków wraz z ich podpisami (klasami),
oraz określenie jak wyglądać ma wygenerowany obrazek, jaki rozmiar itd."""
demo_loader = torch.utils.data.DataLoader(train_set, batch_size=10)

batch = next(iter(demo_loader))
images, labels = batch
print(type(images), type(labels))
print(images.shape, labels.shape)
"""Przycinanie danych wejściowych do prawidłowego zakresu dla imshow z danymi RGB"""
grid = torchvision.utils.make_grid(images, nrow=10)

plt.figure(figsize=(15, 20))
plt.imshow(np.transpose(grid, (1, 2, 0)))
print("labels: ", end=" ")
for i, label in enumerate(labels):
    print(output_label(label), end=", ")
"""Pokazanie na ekranie"""
plt.show()

"""Utworzenie klasy modelowej FashionCNN,dziedziczy klasę z nn.Module, 
będącą superklasą dla wszystkich sieci nauronowych w Pytorch"""
class FashionCNN(nn.Module):

    def __init__(self):
        super(FashionCNN, self).__init__()
        """
        Ta sieć neuronowa ma 2 warstwy.
        
        Conv2d: ("Stosuje splot 2D dla sygnału wejściowego złożonego z kilku płaszczyzn wejściowych")
            in_channels - Liczba kanałów w obrazie wejściowym
            out_channels - Liczba kanałów utworzonych przez splot (convolution)
            kernel_size - Rozmiar jądra
            padding - wypełnienie (padding) dodane to wszystkich 4 stron wejścia.
            
        BatchNorm2d: ("tosuje normalizację wsadową na wejściu 4D (mini-partia wejść 2D z dodatkowym wymiarem kanału")
        
        ReLU: ("Stosuje elementową funkcję wyprostowanej jednostki liniowej")
        
        MaxPool2d: ("Stosuje pulę 2D max dla sygnału wejściowego złożonego z kilku płaszczyzn wejściowych.")
        
        """
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        """Linear - "Stosuje transformację liniową do przychodzących danych"""""
        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        """Dropout2d - Wyzerowywuje losowe kanały z prawdpodobieństwem 0.25"""
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out

"""Utworzenie obiektu / modelu z klasy FashionCNN"""
model = FashionCNN()
"""Transfer na GPU jeśli możliwe"""
model.to(device)
"""Definiowanie funkcji straty. używamy tutaj CrossEntropyLoss() - To kryterium oblicza krzyżową utratę entropii między logami wejściowymi a celem."""
error = nn.CrossEntropyLoss()

learning_rate = 0.001
"""Wykorzystanie algorytmu Adama do celów optymalizacji"""
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
"""Wypisanie na ekranie"""
print(model)

"""Uczenie sieci i testowanie jej na testowym zbiorze danych"""
num_epochs = 5
count = 0
"""Listy do wizualizacji strat i dokładności"""
loss_list = []
iteration_list = []
accuracy_list = []

"""Listy do poznania dokładności klasowej"""
predictions_list = []
labels_list = []

for epoch in range(num_epochs):
    for images, labels in train_loader:
        """Przesyłanie obrazów i etykiet do GPU, jeśli jest dostępne"""
        images, labels = images.to(device), labels.to(device)

        train = Variable(images.view(100, 1, 28, 28))
        labels = Variable(labels)

        # Forward pass
        outputs = model(train)
        loss = error(outputs, labels)

        """Inicjowanie gradientu jako 0, aby nie było mieszania gradientu między partiami"""
        optimizer.zero_grad()

        """Propagacja błędu wstecz"""
        loss.backward()

        # Optimizing the parameters
        optimizer.step()

        count += 1

        """Testowanie modelu"""

        if not (count % 50):  # It's same as "if count % 50 == 0"
            total = 0
            correct = 0

            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels_list.append(labels)

                test = Variable(images.view(100, 1, 28, 28))

                outputs = model(test)

                predictions = torch.max(outputs, 1)[1].to(device)
                predictions_list.append(predictions)
                correct += (predictions == labels).sum()

                total += len(labels)

            accuracy = correct * 100 / total
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)

        if not (count % 500):
            print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))

"""Wizualizacja strat i dokładności za pomocą iteracji,
Pokazanie wykresu i określnie jego nazwy, nazwy obu osi.
"""
plt.plot(iteration_list, loss_list)
plt.xlabel("No. of Iteration")
plt.ylabel("Loss")
plt.title("Iterations vs Loss")
plt.show()

"""
Iteracje vs Skuteczność
Pokazanie wykresu i określnie jego nazwy, nazwy obu osi."""
plt.plot(iteration_list, accuracy_list)
plt.xlabel("No. of Iteration")
plt.ylabel("Accuracy")
plt.title("Iterations vs Accuracy")

plt.show()
"""Ile prawidłowych"""
class_correct = [0. for _ in range(10)]
total_correct = [0. for _ in range(10)]

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        test = Variable(images)
        outputs = model(test)
        predicted = torch.max(outputs, 1)[1]
        c = (predicted == labels).squeeze()

        for i in range(100):
            label = labels[i]
            class_correct[label] += c[i].item()
            total_correct[label] += 1
"""Wypisanie na ekranie dokładnośći dla testowanych przedmiotów"""
for i in range(10):
    print("Accuracy of {}: {:.2f}%".format(output_label(i), class_correct[i] * 100 / total_correct[i]))

"""Drukowanie matrycy zamieszania - confisuion matrix"""
predictions_l = [predictions_list[i].tolist() for i in range(len(predictions_list))]
labels_l = [labels_list[i].tolist() for i in range(len(labels_list))]
predictions_l = list(chain.from_iterable(predictions_l))
labels_l = list(chain.from_iterable(labels_l))

confusion_matrix(labels_l, predictions_l)
print("Classification report for CNN :\n%s\n"
      % (metrics.classification_report(labels_l, predictions_l)))