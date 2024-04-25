import torch
import numpy as np
from torch.autograd import Variable

def calibration_error(model, data_loader,  temperature=0, bias=1, num_bins=10, num_classes=100):
    """
    Calculate the relative calibration error (RCE), maximum calibration error (MCE) and adaptive expected calibration erro (AdaECE) of a PyTorch multi-class
    classification model.

    Args:
        model: PyTorch model object with a forward method that returns logits.
        data_loader: PyTorch data loader object that generates batches of (inputs, targets).
        num_bins: Number of bins to use for histogram.
        num_classes: Number of classes in the classification problem.

    Returns:
        Tuple containing the ECE and MCE as float values.
    """

    # Set model to evaluation mode
    model.eval()

    # Initialize arrays to hold predicted probabilities and true labels
    all_probs = []
    all_labels = []

    # Generate predictions and true labels for each batch in the data loader
    with torch.no_grad():
        for inputs, targets in data_loader:
            # Forward pass through the model
            inputs=inputs.cuda()
            targets=Variable(targets).cuda()

            logits = model(inputs)
            logits = logits / (temperature *0.1* torch.norm(logits,p=2,dim=1,keepdim=True) + bias)

            # Apply the softmax function to get the normalized probabilities
            probs = torch.softmax(logits, dim=-1)

            # Append predicted probabilities and true labels to arrays
            all_probs.append(probs.cpu().numpy())
            all_labels.append(targets.cpu().numpy())

    # Flatten arrays and concatenate along axis 0
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    # Calculate predicted labels
    predicted_labels = np.argmax(all_probs, axis=1)

    # Calculate bin edges
    bin_edges = np.linspace(0, 1, num_bins+1)

    # Initialize arrays to hold bin accuracies and bin confidences for each class
#    max_probs = np.max(all_probs, axis=1)
#    bin_indices = np.digitize(max_probs, bin_edges[1:-1], right=True)

    # Initialize arrays to hold bin accuracies and bin confidences for each class
#    bin_accs = np.zeros((num_classes, num_bins))
#    bin_confs = np.zeros((num_classes, num_bins))

    # Calculate accuracy and average confidence for each bin and class
#    for c in range(num_classes):
#        for i in range(num_bins):
#            bin_idx = np.logical_and(bin_indices == i, all_labels == c)
#            if np.sum(bin_idx) > 0:
#                bin_accs[c, i] = np.mean(predicted_labels[bin_idx] == c)
#                bin_confs[c, i] = np.mean(max_probs[bin_idx])
#            else:
#                bin_accs[c, i] = 0
#                bin_confs[c, i] = 0

    max_probs = np.max(all_probs, axis=1)
    bin_indices = np.digitize(max_probs, bin_edges[1:-1], right=True)
    # Initialize arrays to hold bin accuracies and bin confidences
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    # Calculate accuracy and average confidence for each bin
    for i in range(num_bins):
        bin_idx = bin_indices == i
        if np.sum(bin_idx) > 0:
            bin_accs[i] = np.mean(predicted_labels[bin_idx] == all_labels[bin_idx])
            bin_confs[i] = np.mean(max_probs[bin_idx])
            bin_sizes[i] = np.sum(bin_idx)

        else:
            bin_accs[i] = 0
            bin_confs[i] = 0
            bin_sizes[i] = 0


    # Calculate ECE for each class and average over all classes

    bin_sizes = bin_sizes / np.sum(bin_sizes)
    rce = np.sum((bin_confs - bin_accs)*bin_sizes)
    #ece /= num_classes

    # Calculate MCE for each class and take maximum over all classes
    mce = np.max(np.abs(bin_confs - bin_accs))

    ## Calculate adaECE
    sorted_indices = np.argsort(max_probs)
    bin_size = len(max_probs) // num_bins
    adaece = 0.0
    for bin_idx in range(num_bins):
        start_idx = bin_idx * bin_size
        end_idx = start_idx + bin_size
        if bin_idx == num_bins - 1:
            end_idx = len(max_probs)  # 最后一个分箱可能包含剩余的样本

        bin_prob_mean = np.mean(max_probs[sorted_indices[start_idx:end_idx]])
        bin_true_sum = np.mean(predicted_labels[sorted_indices[start_idx:end_idx]] == all_labels[sorted_indices[start_idx:end_idx]])

        adaece += (abs((end_idx - start_idx) / len(max_probs)) *
                   abs(bin_prob_mean - bin_true_sum ))



    return rce, mce,adaece