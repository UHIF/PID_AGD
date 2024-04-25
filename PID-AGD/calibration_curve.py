import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable



def calibration_curve(model, data_loader, temperature=0, bias=1, num_bins=10, num_classes=10):
    """
    Generate the calibration curve of a PyTorch multi-class classification model.

    Args:
        model: PyTorch model object with a forward method that returns logits.
        data_loader: PyTorch data loader object that generates batches of (inputs, targets).
        num_bins: Number of bins to use for histogram.
        num_classes: Number of classes in the classification problem.

    Returns:
        None. The function generates a plot of the calibration curve.
    """

    # Set model to evaluation mode
    model.eval()

    # Initialize arrays to hold predicted probabilities and true labels
    all_probs = []
    all_labels = []

    # Generate predictions and true labels for each batch in the data loader
    with torch.no_grad():
        for inputs, targets in data_loader:
            # Forward pass through the model8
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

    # Calculate bin edges based on the maximum prediction probability
    bin_edges = np.linspace(0, 1, num_bins+1)
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


    # Average the confidence curves for all classes
    avg_bin_confs = bin_confs
    avg_bin_accs = bin_accs

    # Calculate bin widths
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    # Calculate bin widths
    bin_sizes = bin_sizes / np.sum(bin_sizes)

    # Initialize figure and axes objects
    fig, ax = plt.subplots(figsize=(8, 8))
    # avg_bin_accs[0]=0.12
    # Plot the bar chart for the average confidence curve
    ax.bar(bin_edges[:-1], avg_bin_confs, width=bin_widths, edgecolor='#CC3333', linestyle='-', align='edge', color='#CC0033', alpha=0.9, label='Confidence')
    #ax.bar(bin_edges[:-1], avg_bin_confs, width=bin_widths, edgecolor='black', linestyle='-', align='edge',
    #       color='white', alpha=0.9, label='Confidence',hatch= '/')
    # Plot the red bars for the gap between the confidence curve and the diagonal line
    # Add bar chart for the gap
    ax.bar(bin_edges[:-1], avg_bin_accs, width=bin_widths, edgecolor='#003366', linestyle='-', align='edge', color='#003399', alpha=0.8, label='Accuracy')
    #ax.bar(bin_edges[:-1], avg_bin_accs, width=bin_widths, edgecolor='black', linestyle='-', align='edge', color='white', alpha=0.8, label='Accuracy',hatch='.')

    # Add diagonal line to indicate perfect calibration
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    # Set axis labels and limits
    ax.set_xlabel('Confidence', fontsize=20)
    ax.set_ylabel('Accuracy', fontsize=20)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid()
    ax.legend(prop={'size':20})
    # Show plot



    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.bar(bin_edges[:-1], bin_sizes, width=bin_widths, edgecolor='black', linestyle='-', align='edge', color='#CC0033', alpha=0.9, label='% of Samples')
    print(bin_sizes)
    ax2.set_ylabel('% of Samples', fontsize=20)
    ax2.grid()
    ax2.axvline(np.sum(bin_confs*bin_sizes), linewidth =3,linestyle='--', color='gray', label='Avg. Confidence')
    ax2.text(np.sum(bin_confs*bin_sizes)-0.01, 0.7, 'Avg. Confidence', fontsize=20, color='black', ha='center', va='center', rotation=90, zorder=10)
    ax2.axvline(np.sum(bin_accs*bin_sizes), linestyle='--', linewidth =3, color='gray', label='Accuracy')
    ax2.text(np.sum(bin_accs*bin_sizes)-0.01, 0.3, 'Accuracy', fontsize=20, color='black', ha='center', va='center', rotation=90, zorder=10)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax.legend(prop={'size':20})
    plt.show()

