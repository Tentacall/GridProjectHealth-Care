from utils.dataloader  import load_data
import time
from utils.confLoader import *
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
# model imports
from models.model_v1 import ClassicalModel
from models.model_v1 import QuantamModel
from models.model_v2 import RetinopathyClassification

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
class OrdinalRegressionLoss(nn.Module):
    def __init__(self, num_classes):
        super(OrdinalRegressionLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, predictions, targets):
        # Convert severity targets to cumulative logits
        cum_logits = torch.zeros_like(predictions)
        for i in range(self.num_classes):
            cum_logits[:, i] = torch.sum(targets >= i, dim=1)
        
        # Calculate the negative log-likelihood loss
        loss = nn.functional.cross_entropy(predictions, cum_logits)
        return loss

class Trainner:
    def __init__(self, model, epoch = 1):
        loader = load_data(train_labels_path, test_labels_path, train_image_path, test_image_path, columns, itype = '.jpg', batch_size = 16, shuffle=True, do_random_crop = False, device = 'cpu')
        self.train_loader, self.test_loader, self.valid_loader = loader.create_loader()
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=0.1, momentum = 0.9, weight_decay = 0.0001)
        self.total_loss = 0.0
        self.epoch = epoch

    @staticmethod
    def loading_bar( current_value, total_value, bar_length=40):
        progress = min(1.0, current_value / total_value)
        arrow = '■' * int(progress * bar_length)
        spaces = ' ' * (bar_length - len(arrow))
        print(f'\r[{arrow}{spaces}] {int(progress * 100)}%', end='', flush=True)

    def train(self):
        start_time = time.time()
        for epoch in range(self.epoch):
            running_loss = 0.0
            for i, data in enumerate(self.train_loader):
                inputs, labels = data['image'], data['label']
                inputs, labels = inputs.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                # print(outputs)
                # print(inputs.size(), outputs.size(), labels.size())
                # loss = F.cross_entropy(outputs, labels)
                # loss = self.ordinal_loss(outputs, labels)
                # labels = labels.unsqueeze(1)
                # labels = labels.float()
                # labels = labels / 4
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                self.loading_bar(i, 24)
            print(f"Epoch {epoch+1}, Loss: {running_loss / len(self.train_loader)}")
        delta = time.time() - start_time
        print(f"Finished training in {delta:0.6f} sec")
        print("Total Loss : ", self.total_loss)

    def test(self):
        total = 0
        correct = 0
        soft_correct = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                inputs, labels = data['image'], data['label']
                inputs, labels = inputs.to(device), labels.to(device)
                # labels = labels.float() / 4
                output = self.model(inputs)
                # output = output.squeeze(dim=1)
                # print(output, labels)
                # differences = torch.abs(output - labels)
                # print(differences)
                # for dif in differences:
                #     if dif.item() < 0.125 : correct += 1
                #     if dif.item() < 0.2 : soft_correct += 1
                # total += labels.size(0)
                # print(f"total: {total}, correct = {correct}, soft_correct = {soft_correct}")
                # print(output.data)
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
    
    @staticmethod
    def ordinal_accuracy(y_pred, y_true, tolerance=1):
        correct_predictions = torch.abs(torch.argmax(y_pred, dim=1) - y_true) <= tolerance
        accuracy = correct_predictions.sum().item() / len(y_true)
        return accuracy

    @staticmethod
    def ordinal_loss(y_pred, y_true, num_classes=5):
        loss = 0
        
        # _, y_pred = torch.max(y_pred, 1)
        # y_true.squeeze() # [[16]], [16]
        for i in range(num_classes - 1):
            loss += torch.log(torch.exp(y_pred[ i]) + 1e-10).sum() - y_pred[ i + 1].sum()
            loss *= (y_true > i).float()
        loss += torch.log(torch.exp(y_pred[ num_classes - 1]) + 1e-10).sum()
        return -loss.mean()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    model = ResidualNet("ImageNet", 101, 5, "BAM")
    print(model)
    trainer = Trainner(model, 20)
    trainer.loadModel("model_resnet_v1.2")
    trainer.train()
    trainer.save("model_resnet_v1.3")
    trainer.test()
