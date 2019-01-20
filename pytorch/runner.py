from models.cnn import Cnn
from utils.preprocess import load_data
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn as nn
import os

if __name__ == "__main__":
    trainloader, testloader, classes = load_data("./data", batch_size=100)

    cnn = Cnn()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

    epochs = 1
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = cnn(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()  # update parameters

            running_loss += loss.item()

            if i % 100 == 99:
                loss_info = {
                    "epoch": epoch + 1,
                    "iteration": i + 1,
                    "loss": running_loss / 100,
                }
                print(
                    "epoch:{epoch}, iteration:{iteration}, loss:{loss}".format(
                        **loss_info
                    )
                )
                running_loss = 0.0

    print("Finish training")

    #save model
    model_dir = os.path.join("results",datetime.today().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(model_dir, exist_ok=True)
    torch.save(cnn.state_dict(), os.path.join(model_dir, "model.pickle"))

    correct_num, total_num = 0, 0

    with torch.no_grad():
        for (images, labels) in testloader:
            outputs = cnn(images)
            _, predicted = torch.max(outputs.data, 1)
            total_num += labels.size(0)
            correct_num += (predicted == labels).sum().item()

    print('Accuracy: {:.2f} %%'.format(100 * float(correct_num/total_num)))
