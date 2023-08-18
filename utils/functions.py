import torch

def ordinal_accuracy(y_pred, y_true, tolerance=1):
    correct_predictions = torch.abs(torch.argmax(y_pred, dim=1) - y_true) <= tolerance
    accuracy = correct_predictions.sum().item() / len(y_true)
    return accuracy

def ordinal_loss(y_pred, y_true, num_classes=5):
    loss = 0
    y_true.squeeze() # [[16]], [16]
    for i in range(num_classes - 1):
        loss += torch.log(torch.exp(y_pred[:, i]) + 1e-10).sum() - y_pred[:, i + 1].sum()
        loss *= (y_true > i).float()
    loss += torch.log(torch.exp(y_pred[:, num_classes - 1]) + 1e-10).sum()
    return -loss.mean()