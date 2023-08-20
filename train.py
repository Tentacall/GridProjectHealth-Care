import time
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import yaml
import argparse
import sys
# model imports
from models.model_v1 import ClassicalModel
from models.model_v1 import QuantamModel
from models.model_v2 import RetinopathyClassification
from models.Q_model import QClassifier
from models.Q_model_simple import SimpleQClassifier
from models.Classic_model_simple import SimpleClassifier
from utils.confLoader import *
from utils.dataloader  import load_data


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using Device: ", device)


class Trainner:
    def __init__(self, model, epoch = 1, reduced = False, quite = True):
        loader = load_data(train_labels_path, test_labels_path, train_image_path, test_image_path, columns, itype = '.jpg', batch_size = 16, shuffle=True, do_random_crop = False, device = 'cpu', reduce = reduced)
        self.train_loader, self.test_loader, self.valid_loader = loader.create_loader()
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=0.001, momentum = 0.9, weight_decay = 0.0001)
        self.total_loss = 0.0
        self.epoch = epoch
        self.quite = quite

    @staticmethod
    def loading_bar( current_value, total_value, bar_length=40):
        progress = min(1.0, current_value / total_value)
        arrow = '■' * int(progress * bar_length)
        spaces = ' ' * (bar_length - len(arrow))
        print(f'\r[{arrow}{spaces}] {int(progress * 100)}%', end='', flush=True)

    def train(self, save_checkpoint, filename):
        for epoch in range(self.epoch):
            running_loss = 0.0
            for i, data in enumerate(self.train_loader):
                start_time = time.time()
                inputs, labels = data['image'], data['label']
                inputs, labels = inputs.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                # print(outputs, labels)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                delta = time.time() - start_time
                if save_checkpoint:   
                    self.save(f"{filename}.{i}")
                if not self.quite:
                    self.loading_bar(i, 24)
                    print(f"{i+1}th batch in {delta:0.6f} sec, with loss = {loss}")
            print(f"Epoch {epoch+1}, Loss: {running_loss / len(self.train_loader)}")
        print("Total Loss : ", self.total_loss)

    def test(self):
        y_pred = torch.empty((25,4))
        y_true = torch.empty((25,4))
        count = 0
        total = 0
        correct = 0
        soft_correct = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                inputs, labels = data['image'], data['label']
                inputs, labels = inputs.to(device), labels.to(device)
                # labels = labels.float() / 4
                output = self.model(inputs)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                y_pred[count], y_true[count] = predicted, labels
                count += 1
                if not self.quite:
                    print(predicted, labels)
                print(f"total: {total}, correct = {correct}")
        print(f'Accuracy of the network on the {total} test images: {100 * correct / total:.2f}%')
        print(y_pred, y_true)
        return y_pred, y_true

    def conf_mat(self, num_of_classes=5):
        y_pred, y_true = self.test()
        # Flatten the tensors
        y_pred_flat = y_pred.flatten()
        y_true_flat = y_true.flatten()
        # print(y_pred_flat.size(), y_true_flat.size())
    
        # Calculate the confusion matrix
        cm = confusion_matrix(y_true_flat, y_pred_flat)
        
        # Calculate recall, precision, and f1-score using the classification report
        class_names = [str(i) for i in range(num_of_classes)]  # Change this if you have meaningful class labels
        report = classification_report(y_true_flat, y_pred_flat, target_names=class_names, output_dict=True)
    
        # Print the classification report
        for class_name, metrics in report.items():
            if class_name in class_names:
                print(f"Metrics for Class {class_name}:")
                print(f"Recall: {metrics['recall']:.4f}")
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"F1-score: {metrics['f1-score']:.4f}")
                print()
    
        # Plot the confusion matrix
        plt.figure(figsize=(num_of_classes, num_of_classes))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(num_of_classes)
        plt.xticks(tick_marks, range(num_of_classes))
        plt.yticks(tick_marks, range(num_of_classes))
        plt.xlabel("Predicted")
        plt.ylabel("True")
        # plt.show()
        plt.savefig("images/confq_matrix.png")
        plt.close()

    def save(self, name):
        save_path = f"models/checkpoint/{name}.pth"
        torch.save(self.model.state_dict(), save_path)
        print("Model saved in ", save_path, ".pth")

    def loadModel(self, name):
        loadpath = f"models/checkpoint/{name}.pth"
        self.model.load_state_dict(torch.load(loadpath))
        self.model.eval()


if __name__ == '__main__':
    # adding command line arg parser
    parser = argparse.ArgumentParser(description="DR Classification models, training, testing utility")
    parser.add_argument("--config", type=str, default='config/simple_quantam.yaml', help="Configuration file path")
    args = parser.parse_args()
    
    models = [ "QuantamModel", "QClassifier", "SimpleQClassifier" ]
    conf = yaml.safe_load(open(args.config, 'r'))
    
    model_name = conf['config']['model']
    reduced = bool(conf['config']['reduced'])
    epoch = int(conf['config']['epoch'])
    preload = bool(conf['config']['preload'])
    preload_path = str(conf['config']['preload'])
    train = bool(conf['config']['train'])
    test = bool(conf['config']['test'])
    save = bool(conf['config']['save'])
    save_path = str(conf['config']['save_path'])
    confusion = bool(conf['config']['confusion_matrix'])
    save_check_point = bool(conf['config']['save_checkpoint'])
    quite = bool(conf['config']['quite'])
    
    if model_name not in models:
        print("Wrong model name. Try these :")
        print(model_name)
        sys.exit(0)
    
    if model_name == "QuantamModel":
        model = QuantamModel(device)
        trainer = Trainner(model, epoch, False , quite)
        
    elif model_name == "QClassifier":
        model = QClassifier(device)
        trainer = Trainner(model, epoch, False , quite)
        
    elif model_name == "SimpleQClassifier":
        model = SimpleQClassifier(device)
        trainer = Trainner(model, epoch, True , quite)
    
    mp.set_start_method('spawn')
    print(model)
    
    if preload:
        trainer.loadModel(preload_path)
    
    if train:
        trainer.train(save_check_point, save_path)
        
    if save:
        trainer.save(save_path)
    
    if test:
        trainer.test()
        
    if confusion:
        trainer.conf_mat()