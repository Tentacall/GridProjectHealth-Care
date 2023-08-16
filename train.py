from utils.dataloader  import load_data
import time
from utils.confLoader import *
import torch.nn as nn
import torch.optim as optim
import torch
# model imports
from models.model_v1 import ClassicalModel
from models.model_v1 import QuantamModel
import torch.multiprocessing as mp

class Trainner:
    def __init__(self, model, epoch = 1):
        loader = load_data(train_labels_path, test_labels_path, train_image_path, test_image_path, columns, itype = '.jpg', batch_size = 16, shuffle=True, do_random_crop = False, device = 'cpu')
        self.train_loader, self.test_loader, self.valid_loader = loader.create_loader()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        self.total_loss = 0.0
        self.epoch = epoch

    def loading_bar(self, current_value, total_value, bar_length=40):
        progress = min(1.0, current_value / total_value)
        arrow = '■' * int(progress * bar_length)
        spaces = ' ' * (bar_length - len(arrow))
        print(f'\r[{arrow}{spaces}] {int(progress * 100)}%', end='', flush=True)

    def train(self):
        start_time = time.time()
        for epoch in range(self.epoch):
            print(f"Running Epoch no: {epoch+1}")
            for i, data in enumerate(self.train_loader):
                inputs, labels = data['image'], data['label']
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                # print(inputs.size(), outputs.size(), labels.size())
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.total_loss += loss.item()
                self.loading_bar(i, 24)
        delta = time.time() - start_time
        print(f"Finished training in {delta:0.6f} sec")
        print("Total Loss : ", self.total_loss)

    def test(self):
        total = 0
        correct = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                inputs, labels = data['image'], data['label']
                output = self.model(inputs)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                print(predicted, labels)
                correct += (predicted == labels).sum().item()
                print(f"total: {total}, correct = {correct}")
        print(f'Accuracy of the network on the {total} test images: {100 * correct / total:.2f}%')
    
    def showStats(self):
        # : TODO 
        pass

    def save(self, name):
        save_path = f"models/checkpoint/{name}.pth"
        torch.save(self.model.state_dict(), save_path)
        print("Model saved in ", save_path, ".pth")

    def loadModel(self, name):
        loadpath = f"models/checkpoint/{name}.pth"
        self.model.load_state_dict(torch.load(loadpath))
        self.model.eval()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    trainer = Trainner(QuantamModel(), 1)
    # trainer.train()
    # trainer.save("model_quant_v1.0")
    trainer.loadModel("model_quant_v1.0")
    trainer.test()

#-------------------------------------------------#
# ClassicalModel -> 31.37% [ 1 epoch ] | problem -> constant result
# Hybreed Model -> 0% [ 1 epoch ] | -> constant result
