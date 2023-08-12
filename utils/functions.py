def accuracy_util(y_pred, y_true):
    difference = abs(y_pred - y_true)

    if difference == 0:
        accuracy_score = 4
    elif difference == 1:
        accuracy_score = 3
    elif difference == 2:
        accuracy_score = 2
    elif difference == 3:
        accuracy_score = 1
    else:
        accuracy_score = 0
        
    accuracy_score = accuracy_score/4

    return accuracy_score

def batch_accuracy(y_pred_batch, y_true_batch):
    
    correct = 0
    
    for i in range(len(y_pred_batch)):
        correct = correct + accuracy_util(y_pred_batch[i], y_true_batch[i])
        correct = correct/len(y_pred_batch)
        
    return correct