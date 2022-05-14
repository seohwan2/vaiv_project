# %%
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

learning_rate = 0.001
training_epochs = 10
batch_size = 100

mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True, 
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)


data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test,
                                          batch_size=batch_size,
                                          shuffle=False)
                                        

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = torch.nn.Linear(3136, 1000, bias=True)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(1000, 10, bias=True)

    

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


model = CNN().to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)    # Softmax is internally computed.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(data_loader)
print('Learning started. It takes sometime.')
i = 1
x =[]
y =[]
for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        loss = criterion(hypothesis, Y)
        loss.backward()
        optimizer.step()

        avg_cost += loss / total_batch
        if i % 1000 == 0:
            x.append(i)
            y.append(loss.item())
        i+=1
    print('[Epoch: {:>4}] loss = {:>.9}'.format(epoch + 1, avg_cost))
print('Learning Finished!')
# %%
with torch.no_grad():
    same = 0
    num = 1
    pred = []
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        prediction = output.data.max(1)[1]
        pred.append(prediction)
        same += prediction.eq(target.data).sum()
        num +=1
    accuracy = (100. * same / len(test_loader.dataset))
    print('Accuracy: ', accuracy, prediction)
# %%
plt.plot(x, y, 'g')
plt.xlabel('Train')
plt.ylabel('Loss')
plt.show()

plt.figure(figsize=(42,7))

title = 'Accuracy : {0}%' .format(accuracy)
plt.suptitle(title, size = 30)

for i in range(10,20):
    plt.subplot(1, 10, i-9)
    plt.imshow(mnist_test[i][0][0], cmap = 'binary')
    if(mnist_test[i][1] == pred[0][i].item()):
        str = 'Label {0}, Predict {0} => Correct!'.format(mnist_test[i][1]).format(pred[0][i].item())
    else:
        str = 'Label {0}, Predict {0} => Not correct!'.format(mnist_test[i][1]).format(pred[0][i].item())
    plt.subplots_adjust(wspace=0.1)
    plt.title(str)  
