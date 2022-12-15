import numpy as np
import torch as T
"""
* klasyfikacja autentyczności banknotów BNN *

Dataset: https://archive.ics.uci.edu/ml/datasets/banknote+authentication

Autorzy:
- Bartosz Krystowski s19545
- Robert Brzoskowski s21162

Przygotowanie środowiska:
Instalacja bibliotek: pandas, numpy
"""
"""CPU jako domyslne urządzenie do obliczeń"""
device = T.device("cpu")  # apply to Tensor or Module


#        1         2         3         4         5         6
# 3456789012345678901234567890123456789012345678901234567890
# ----------------------------------------------------------
# predictors and label in same file
# archive.ics.uci.edu/ml/datasets/banknote+authentication
# IDs 0001 to 1372 added
# data has been k=20 normalized (all four columns)
# ID  variance  skewness  kurtosis  entropy  class
# [0]    [1]      [2]       [3]       [4]     [5]
#  (0 = authentic, 1 = forgery)  # verified
# train: 1097 items (80%), test: 275 item (20%)

class BanknoteDataset(T.utils.data.Dataset):
    """Metoda __init__() rozpoczyna się od wczytania wszystkich odpowiednich danych z pliku do pamięci za pomocą funkcji NumPy loadtxt():
    Dane są wczytywane do macierzy NumPy jako wartości typu float32
    """
    def __init__(self, src_file, num_rows=None):
        all_data = np.loadtxt(src_file, max_rows=num_rows,
                              usecols=range(1, 6), delimiter="\t", skiprows=0,
                              dtype=np.float32)  # strip IDs off
        """Po wczytaniu wszystkich danych do pamięci wiersze predyktorów są wyodrębniane, 
        a następnie konwertowane na tensory PyTorch:"""
        self.x_data = T.tensor(all_data[:, 0:4],
                               dtype=T.float32).to(device)
        self.y_data = T.tensor(all_data[:, 4],
                               dtype=T.float32).to(device)

        self.y_data = self.y_data.reshape(-1, 1)
    """Metoda __len__ zwracająca rzeczywistą liczbę wierszy odczytanych danych"""
    def __len__(self):
        return len(self.x_data)

    """Metoda __getitem__ zwraca jeden lub więcej elementów danych jako obiekt słownika, 
    w którym klucz „predyktorów” podaje wartości x predyktora, 
    a klucz „target” daje wartości docelowe 0 lub 1.
    Metoda zaczyna się od sprawdzenia parametru idx, aby zobaczyć, czy jest to tensor PyTorch, a jeśli tak, konwertuje tensor na zwykłą listę. """
    def __getitem__(self, idx):
        if T.is_tensor(idx):
            idx = idx.tolist()
        preds = self.x_data[idx, :]  # idx rows, all 4 cols
        lbl = self.y_data[idx, :]  # idx rows, the 1 col
        sample = {'predictors': preds, 'target': lbl}

        return sample


# ---------------------------------------------------------

def accuracy(model, ds):
    """ds jest iterowalnym zestawem danych tensorów"""
    n_correct = 0
    n_wrong = 0

    """utwórz DataLoader, a następnie wylicz go"""
    for i in range(len(ds)):
        inpts = ds[i]['predictors']
        target = ds[i]['target']  # float32  [0.0] or [1.0]
        with T.no_grad():
            oupt = model(inpts)

        # avoid 'target == 1.0'
        if target < 0.5 and oupt < 0.5:  # .item() not needed
            n_correct += 1
        elif target >= 0.5 and oupt >= 0.5:
            n_correct += 1
        else:
            n_wrong += 1

    return (n_correct * 1.0) / (n_correct + n_wrong)


# ---------------------------------------------------------

def acc_coarse(model, ds):
    inpts = ds[:]['predictors']  # all rows
    targets = ds[:]['target']  # all target 0s and 1s
    with T.no_grad():
        oupts = model(inpts)  # all computed ouputs
    pred_y = oupts >= 0.5  # tensor of 0s and 1s
    num_correct = T.sum(targets == pred_y)
    acc = (num_correct.item() * 1.0 / len(ds))  # scalar
    return acc


# ----------------------------------------------------------

def my_bce(model, batch):
    sum = 0.0
    inpts = batch['predictors']
    targets = batch['target']
    with T.no_grad():
        oupts = model(inpts)
    for i in range(len(inpts)):
        oupt = oupts[i]
        # should prevent log(0) which is -infinity
        if targets[i] >= 0.5:  # avoiding == 1.0
            sum += T.log(oupt)
        else:
            sum += T.log(1 - oupt)

    return -sum / len(inpts)


# ----------------------------------------------------------

class Net(T.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hid1 = T.nn.Linear(4, 8)  # 4-(8-8)-1
        self.hid2 = T.nn.Linear(8, 8)
        self.oupt = T.nn.Linear(8, 1)

        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.hid2.weight)
        T.nn.init.zeros_(self.hid2.bias)
        T.nn.init.xavier_uniform_(self.oupt.weight)
        T.nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        z = T.tanh(self.hid1(x))
        z = T.tanh(self.hid2(z))
        z = T.sigmoid(self.oupt(z))
        return z


# ----------------------------------------------------------

def main():
    print("\nBanknote authentication using PyTorch \n")
    T.manual_seed(1)
    np.random.seed(1)

    """tworzy obiekty Dataset i DataLoader"""
    print("Creating Banknote train and test DataLoader ")

    train_file = ".\\Data\\banknote_k20_train.txt"
    test_file = ".\\Data\\banknote_k20_test.txt"

    train_ds = BanknoteDataset(train_file)  # all rows
    test_ds = BanknoteDataset(test_file)

    bat_size = 10
    train_ldr = T.utils.data.DataLoader(train_ds,
                                        batch_size=bat_size, shuffle=True)

    """tworzenie sieci neuronowej"""
    print("Creating 4-(8-8)-1 binary NN classifier ")
    net = Net().to(device)

    """Trening"""
    print("\nPreparing training")
    net = net.train()  # set training mode
    lrn_rate = 0.01
    loss_obj = T.nn.BCELoss()  # binary cross entropy
    optimizer = T.optim.SGD(net.parameters(),
                            lr=lrn_rate)
    max_epochs = 100
    ep_log_interval = 10
    print("Loss function: " + str(loss_obj))
    print("Optimizer: SGD")
    print("Learn rate: 0.01")
    print("Batch size: 10")
    print("Max epochs: " + str(max_epochs))

    print("\nStarting training")
    for epoch in range(0, max_epochs):
        epoch_loss = 0.0  # for one full epoch
        epoch_loss_custom = 0.0
        num_lines_read = 0

        for (batch_idx, batch) in enumerate(train_ldr):
            X = batch['predictors']  # [10,4]  inputs
            Y = batch['target']  # [10,1]  targets
            oupt = net(X)  # [10,1]  computeds

            loss_val = loss_obj(oupt, Y)  # a tensor
            epoch_loss += loss_val.item()  # accumulate

            optimizer.zero_grad()  # reset all gradients
            loss_val.backward()  # compute all gradients
            optimizer.step()  # update all weights

        if epoch % ep_log_interval == 0:
            print("epoch = %4d   loss = %0.4f" % \
                  (epoch, epoch_loss))

    print("Done ")

    # ----------------------------------------------------------

    """Ocena modelu"""
    net = net.eval()
    acc_train = accuracy(net, train_ds)
    print("\nAccuracy on train data = %0.2f%%" % \
          (acc_train * 100))
    acc_test = accuracy(net, test_ds)
    print("Accuracy on test data = %0.2f%%" % \
          (acc_test * 100))

    """Zapis modelu"""
    print("\nSaving trained model state_dict \n")
    path = ".\\Models\\banknote_sd_model.pth"
    T.save(net.state_dict(), path)

    """Wykonanie predykcji"""
    raw_inpt = np.array([[4.4, 1.8, -5.6, 3.2]],
                        dtype=np.float32)
    norm_inpt = raw_inpt / 20
    unknown = T.tensor(norm_inpt,
                       dtype=T.float32).to(device)

    print("Setting normalized inputs to:")
    for x in unknown[0]:
        print("%0.3f " % x, end="")

    net = net.eval()
    with T.no_grad():
        raw_out = net(unknown)  # a Tensor
    pred_prob = raw_out.item()  # scalar, [0.0, 1.0]

    print("\nPrediction prob = %0.6f " % pred_prob)
    if pred_prob < 0.5:
        print("Prediction = authentic")
    else:
        print("Prediction = forgery")

    print("\nEnd Banknote demo")


if __name__ == "__main__":
    main()