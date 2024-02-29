This text file contains the task provided and later details of the implementation (inclduing what each file does)
As a coding task, I would suggest that the following then makes the most sense:

1) Reimplement the gradient accumulation method from this paper: https://arxiv.org/abs/2307.09542. This is supposed to find out which neurons are most influenced by a data point during training.
2) Train a VGG7 model with the MNIST dataset. As done in the paper, assign random labels to 10% of the data points from MNIST. These are the outliers.
3) Then, for the 10% outliers, use gradient accumulation to find out which neurons they have the largest influence to.

4) Take 10% of the normal data point and use gradient accumulation to find which neurons they have the largest influence to.

5) Report and visualize with a visualization method of your choice, how large the overlap between the neurons is. Also see if you can find other interesting insights, like, for example, if the memorized data points are mainly memorized by the same neurons, or all by different ones – what their class has got to do with it. Potentially, it will also make sense to visualize the data points for that.

####### Implementation #######

preparedata.py: contains a function to add noise to the dataset and return the indices of those examples.

train.ipynb:    trains a VGG-7 model on the MNIST dataset containing 60,000 examples of handwritten digits. 
                At the end the resulting gradients of each layer from the last epoch are saved.

train_outliers.ipynb:   trains a VGG-7 model on the MNIST dataset containing 60,000 examples with 10% noisy examples. 
                        Noise: The output has been changed randomly for 6,000 examples. 
                        The model is trained on 54,000 + 6,000 clean and noisy examples.
                        The resulting model is also saved using checkpoints. these are the .ckpt files
                        The indices of noisy samples is saved in "noisy_y_train.csv", "noisy_indices.csv" to continue working on them for locating important neurons.

evaluation_gradients:   The file is incomplete. The point is to track gradients for noisy and clean examples. 
                        Ideally the gradients should follow eachother closely with increasing layer depth. 

get neurons:            extract most important neurons for an example by checking if prediction is flipped when the neuron is set to zero.
                        This is done for 1000 examples of both clean and noisy data.
                        The same function is copied to compute them separately for convolutional layers and fully connected layers due to shape difference.
                        The neurons are saved in a dictionary. The keys are layer names with a list of list as value. Each value contains 1000 lists (datapoint/example), and each datapoint is then a list of neurons it affects.
                        The dictionaries are saved in .json files

evaluate neurons:       here i want evaluate the neurons that are memorizing the data.  
                        datapoints whose prediction are not affected by zeroing out neurons are removed.
                        this is done to maintain the index of the datapoint.

                        next i want to look at the overlap in neurons for the examples and see if their labels have anything to do with that. 









