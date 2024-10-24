import numpy as np

from matplotlib import pyplot as plt
import funweightclust as funweight
from sklearn import metrics as met

def split_fda_chunks(fd1, fd2, num_chunks, labels):
    """
    Description
    -----------
    This function splits functional data into chunks that will be used when testing the model.

    Parameters
    ----------
    fd1: `FDataBasis` or `list` of `FDataBasis`
        a `FDataBasis` object that contains functional data fit to a set of
        basis functions, or a list of these

    fd2: `FDataBasis` or `list` of `FDataBasis`
        a `FDataBasis` object that contains functional data fit to a set of
        basis functions, or a list of these

    num_chunks: 'int'
        The number of chunks you want the data split into. For example splitting the data into 4 smaller chunks of data.

    labels: list of 'int'
        The labels for each entry that notify which cluster it belongs to.
    """
    # Get indices for each cluster
    label_cluster1_indices = np.random.permutation(np.where(labels == 0)[0])
    label_cluster2_indices = np.random.permutation(np.where(labels == 1)[0])

    # Split indices into chunks
    label_cluster1_chunks = np.array_split(label_cluster1_indices, num_chunks)
    label_cluster2_chunks = np.array_split(label_cluster2_indices, num_chunks)



    # Combine chunks from both clusters
    label_chunk_indices = [np.concatenate((label_cluster1_chunks[i], label_cluster2_chunks[i])) for i in range(num_chunks)]
    label_chunks = [labels[label_chunk_indices[i]] for i in range(num_chunks)]
    # Create empty lists for functional data chunks
    chunks_fd1 = []
    chunks_fd2 = []



    # Create chunks for fd1 and fd2 based on the label_chunks
    for chunk in label_chunk_indices:
        chunks_fd1.append(fd1[chunk])
        chunks_fd2.append(fd2[chunk])


    return chunks_fd1, chunks_fd2, label_chunks

def select_and_combine_chunks(fd_chunks, selected_chunk_index):
    """
    Description
    -----------
    This function selects the specified test data from fd_chunks, and flattens the rest into training_data.

    Parameters
    ----------
    fd_chunks: list of 'FDataObject'
        A list of FDataObjects from which one is the testing data and the rest are training data

    selected_chunk_index: 'int'
        The index of the chunk that is the testing data
    """
    # Select the specified chunk
    selected_chunk = fd_chunks[selected_chunk_index]
    
    # Combine the remaining chunks
    remaining_chunks = [chunk for i, chunk in enumerate(fd_chunks) if i != selected_chunk_index]
    combined_remaining_chunks = remaining_chunks[0]
    for chunk in remaining_chunks[1:]:
        combined_remaining_chunks = combined_remaining_chunks.concatenate(chunk)
    
    return selected_chunk, combined_remaining_chunks

def plot_chunks(split_data, split_labels, x_or_y):
    num_chunks = len(split_data)
    # Assuming num_chunks and split_data are already defined
    fig, axes = plt.subplots(1, num_chunks, figsize=(15, 5), sharey=True)  # Create 1 row and num_chunks columns of subplots

    for i in range(num_chunks):
        # Plot each chunk on its corresponding axis and color-code by the labels
        split_data[i].plot(axes=axes[i], group=split_labels[i])  # Pass the axes and group for color coding

        # Set the title for each subplot
        axes[i].set_title(f"Chunk {i+1}")

    # Adjust layout to avoid overlapping of subplots
    plt.tight_layout()
    fig.suptitle(f'{x_or_y} Functional Data Split into Chunks', fontsize=16)

    plt.show()

def test_predict(data, number_of_chunks, threshold, region="cingulum"):
    """
    Description
    -----------
    This function tests the model by using the prediction. It does so by separating the functional data into equal chunks. With these chunks one chunk is chosen as the testing data, and the rest are used to train the model. This process is repeated until each respective chunk has been used as testing data.

    Parameters
    ----------
    number_of_chunks: 'int'
        The number of chunks the Functional Data is to be separated into

    threshold: 'double'
    
    region: 'string'
        Either "cingulum" or "corpus".
    """
    
    fdobj = data['fdx']
    fdobjy = data['fdy']
    labels = data['groupd']

    split_data, split_datay, split_labels = split_fda_chunks(fdobj, fdobjy, number_of_chunks, labels)
    
    plot_chunks(split_data, split_labels, "X")
    plot_chunks(split_datay, split_labels, "Y")

    confusion_matrices = []
    ari_scores = []
    accuracy_rates = []

    for i in range(number_of_chunks):
        test_labels = np.array(split_labels[i])

        test_data, training_data = select_and_combine_chunks(split_data, i)
        test_datay, training_datay = select_and_combine_chunks(split_datay, i)


        training_labels = np.concatenate([split_labels[j] for j in range(len(split_labels)) if j != i])

        print("Training ", i+1, ":")
        res = funweight.funweightclust(training_data, training_datay, K=2, model="all", modely="all", init="kmeans", nb_rep=1, threshold=threshold, known=training_labels, verbose=False)
        p = res.predict(test_data, test_datay)
        if (p['class'] is not None):
            matrix = met.confusion_matrix(test_labels, p['class'])

            if np.trace(matrix) < np.trace(np.fliplr(matrix)):
                matrix = np.fliplr(matrix)

            confusion_matrices.append(matrix)
            ari_scores.append(met.adjusted_rand_score(test_labels, p['class'])) 
            accuracy_rates.append(float(np.trace(matrix) / np.sum(matrix)) * 100)

    return accuracy_rates, ari_scores, confusion_matrices

