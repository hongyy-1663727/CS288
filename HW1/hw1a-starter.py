#!/usr/bin/env python
# coding: utf-8

# ## Project 1a: Intro to PyTorch Mini-Project
# 
# Welcome to CS 288.  The purpose of this first assignment is to make sure that you are familiar with all the tools you need to complete the programming assignments for the course.  We will walk you through the process of building a model with PyTorch in Kaggle.  Most of it will be structured as a tutorial, but we will ask you to fill in code and submit at the end. 
# 
# Please note that this assignment is not representative of the projects for this course - other projects will be considerably more involved and take longer to complete.
# 
# Our assignments will be given to you as Jupyter notebooks, and we intend for you to run them using Kaggle or Google Colab. Kaggle has historically had better availability of GPUs so we recommend using Kaggle.
# 
# ### Kaggle
# First go to https://www.kaggle.com/ and create an account. Click `Create` on the top right click `Copy & Edit` the notebook.
# 
# To access a GPU, first verify your phone number. Then, click the three dots on the right and `Accelerator` in the drop down and select a GPU.
# 
# ### Colab
# To get started, make a copy of the assignment by clicking `File->Save a copy in drive...`.  You will need to be logged into a Google account, such as your @berkeley.edu account.
# 
# To access a GPU, go to `Edit->Notebook settings` and in the `Hardware accelerator` dropdown choose `GPU`. 
# As soon as you run a code cell, you will be connected to a cloud instance with a GPU.
# 
# 
# Try running the code cell below to check that a GPU is connected (select the cell then either click the play button at the top left or press `Ctrl+Enter` or `Shift+Enter`).

# In[7]:


import torch

if torch.backends.mps.is_available():
    mps_device = torch.device('mps')
    print('Found GPU')
else:
    print('Did not find GPU')


# Note for Colab users: When you run a code cell, Colab executes it on a temporary cloud instance.  Every time you open the notebook, you will be assigned a different machine.  All compute state and files saved on the previous machine will be lost.  Therefore, you may need to re-download datasets or rerun code after a reset. If you save output files that you don't want to lose, you should download them to your personal computer before moving on to something else. You can download files by hitting the > arrow at the top left of the page under the menus to expand the sidebar, selecting `Files`, right clicking the file you want, and clicking `Download`.  Alternatively, you can mount your Google drive to the temporary cloud instance's local filesystem using the following code snippet and save files under the specified directory (note that you will have to provide permission every time you run this).

# Many of the assignments will require training a model for some period of time, often on the order of 20-30 minutes.  There are some important limitations to Colab that you should be aware of when running code for this amount of time.  If you close the window or put your computer to sleep, Colab will disconnect you from the compute machine and your code will stop running.   There are also timeouts for inactivity (somewhere on the order of 30 minutes), so if you want to leave code running, be sure to check back periodically.  After a timeout, your compute machine will be disconnected and the files on it will be lost.
# 
# A few other notes about using Colab:
# * The `Runtime` menu has many different run options, such as `Run all` or `Run after` so you don't have to run each code block individually.
# * Some people have run into CUDA device assert errors that did not originate from their code.  Restarting the runtime should fix this (unless there actually is a problem with your code).
# 
# If at some point you want to run longer jobs or connect to multiple GPUs, there are coupons for Google Compute Cloud available for students in the course. You could deploy your own cloud instance and run JupyterHub to recreate a similar environment to Colab. However, the course staff cannot offer technical support for this kind of configuration; you're on your own to set it up.

# ### Part-of-Speech Tagging
# 
# You'll be trying to predict the most common [part of speech](https://web.stanford.edu/~jurafsky/slp3/8.pdf) for a word from its characters.  This project will focus on word types rather than tokens and not use any context (https://en.wikipedia.org/wiki/Type%E2%80%93token_distinction). This task is different from (and simpler than) a standard part-of-speech tagging task, which predicts part-of-speech tags for tokens in their sentential context.
# 
# Many words can have multiple different parts of speech, but in this project we will associate each word only with its most common part of speech in the [Brown Corpus](https://www1.essex.ac.uk/linguistics/external/clmt/w3c/corpus_ling/content/corpora/list/private/brown/brown.html), which has been manually labeled with part-of-speech tags.  
# 
# Words are lowercased and filtered for length and frequency. Punctuation and numbers are removed. Any real NLP application would have to deal with the actual contents of text instead of filtering in this way, but we're just warming up.
# 
# Below, we provide you with code to load the dataset. Please don't change the cell below, or you may confuse our autograder.

# In[8]:


import nltk
import random

from nltk.corpus import brown
from collections import defaultdict, Counter

nltk.download('brown')
nltk.download('universal_tagset')

brown_tokens = brown.tagged_words(tagset='universal')
print('Tagged tokens example: ', brown_tokens[:5])
print('Total # of word tokens:', len(brown_tokens))

max_word_len = 20

def most_common(s):
    "Return the most common element in a sequence."
    return Counter(s).most_common(1)[0][0]

def most_common_tags(tagged_words, min_count=3, max_len=max_word_len):
    "Return a dictionary of the most common tag for each word, filtering a bit."
    counts = defaultdict(list)
    for w, t in tagged_words:
        counts[w.lower()].append(t)
    return {w: most_common(tags) for w, tags in counts.items() if 
            w.isalpha() and len(w) <= max_len and len(tags) >= min_count}

brown_types = most_common_tags(brown_tokens)
print('Tagged types example: ', sorted(brown_types.items())[:5])
print('Total # of word types:', len(brown_types))

def split(items, test_size):
    "Randomly split into train, validation, and test sets with a fixed seed."
    random.Random(288).shuffle(items)
    once, twice = test_size, 2 * test_size
    return items[:-twice], items[-twice:-once], items[-once:]

val_test_size = 1000
all_data_raw = split(sorted(brown_types.items()), val_test_size)
train_data_raw, validation_data_raw, test_data_raw = all_data_raw
all_tags = sorted(set(brown_types.values()))
print('Tag options:', all_tags)


# You're welcome to insert additional cells and explore the data. Our autograders don't rely on any particular structure of the notebook.

# In[ ]:


# Explore the data here as you see fit.


# First, let's run a baseline that predicts `NOUN` for every word. A predictor function takes a list of tagged words and returns a list of predicted tags. We've also provided some helper functions here to evaluate model outputs.  You don't need to fill in any code in this cell.

# In[4]:


def noun_predictor(raw_data):
    "A predictor that always predicts NOUN."
    predictions = []
    for word, _ in raw_data:
        predictions.append('NOUN')
    return predictions

def accuracy(predictions, targets):
    """Return the accuracy percentage of a list of predictions.
    
    predictions has only the predicted tags
    targets has tuples of (word, tag)
    """
    assert len(predictions) == len(targets)
    n_correct = 0
    for predicted_tag, (word, gold_tag) in zip(predictions, targets):
        if predicted_tag == gold_tag:
            n_correct += 1

    return n_correct / len(targets) * 100.0

def evaluate(predictor, raw_data):
    return accuracy(predictor(raw_data), raw_data)

def print_sample_predictions(predictor, raw_data, k=10):
    "Print the first k predictions."
    d = raw_data[:k]
    print('Sample predictions:', 
          [(word, guess) for (word, _), guess in zip(d, predictor(d))])

print('noun baseline validation accuracy:', 
      evaluate(noun_predictor, validation_data_raw))
print_sample_predictions(noun_predictor, validation_data_raw)


# ### Building a PyTorch Classifier
# 
# We will be using the deep learning framework PyTorch for all our projects.
# If you haven't used PyTorch at all before, we recommend you check out the tutorials on the PyTorch website: https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html.  Throughout this project and the others in this course, you will need to reference the documentation at https://pytorch.org/docs/stable/index.html.  We'll be using PyTorch version 1.7, which comes pre-installed with Colab.  In this project, we'll walk you through the process of defining and training your neural network model, but future projects will have less guidance.
# 
# Below, we've provided a baseline network as a PyTorch Module that will learn a single parameter per part-of-speech tag. This model has the capacity to learn that `'NOUN'` is the most common tag and predict that. It can't do better. Use this network as you are developing your training and prediction code, then replace it with your actual network later.

# In[5]:


import torch
from torch import nn
import torch.nn.functional as F

class BaselineNetwork(nn.Module):
    def __init__(self, n_outputs):
        super().__init__()

        # learn a vector of size n_outputs, initialized with all zeros
        self.param = nn.Parameter(torch.zeros(n_outputs)) 

    def forward(self, chars, mask):
        # return the same outputs (self.param) for each example in a batch
        return self.param.expand(chars.size(0), -1)


# To train or evaluate a neural model, we'll need to transform the raw data from strings into tensors.  We've provided the following function to perform the transformation for you. Each word is prepended with the `^` character and appended with `$` so that these boundary characters are available to the network.

# In[6]:


def make_matrices(data_raw):
    """Convert a list of (word, tag) pairs into tensors with appropriate padding.
    
    character_matrix holds character codes for each word, 
      indexed as [word_index, character_index]
    character_mask masks valid characters (1 for valid, 0 invalid), 
      indexed similarly so that all inputs can have a constant length
    pos_labels holds part-of-speech one-hot vectors,
      indexed as [word_index, pos_index] with 0/1 values TODO this is wrong, pos_labels is a vector
    """
    max_len = max_word_len + 2  # leave room for word start/end symbols
    character_matrix = torch.zeros(len(data_raw), max_len, dtype=torch.int64, device=mps_device)
    character_mask = torch.zeros(len(data_raw), max_len, dtype=torch.float32,device=mps_device)
    pos_labels = torch.zeros(len(data_raw), dtype=torch.int64,device=mps_device)
    for word_i, (word, pos) in enumerate(data_raw):
        for char_i, c in enumerate('^' + word + '$'):
            character_matrix[word_i, char_i] = ord(c)
            character_mask[word_i, char_i] = 1
        pos_labels[word_i] = all_tags.index(pos)
    return torch.utils.data.TensorDataset(character_matrix, character_mask, pos_labels)

validation_data = make_matrices(validation_data_raw)

print('Sample datapoint after preprocessing:', validation_data[1])
print('Raw datapoint:', validation_data_raw[1])


# The output of a `BaselineNetwork` is a matrix of dimension (batch_size, num_pos_labels) containing logits, or unnormalized log probabilities. To get probabilities from this matrix, you would run `F.softmax(x, dim=1)`, which exponentiates the logits and then normalizes each row to sum to 1.  The cell below generates an output distribution for the first example of the validation set, which is uniform because the network param was initialized to zero.
# 
# In PyTorch, it is common to return pre-activation values from modules (e.g. the values before running the final softmax or sigmoid operation).  PyTorch has loss functions that combine the softmax/sigmoid operation into the loss operation for more numerical stability.  Be sure you know what type of values a network returns, as this will affect your training and prediction code.

# In[ ]:


# Create a network and copy its parameters to the GPU.

untrained_baseline = BaselineNetwork(len(all_tags))
untrained_baseline.eval()

# Select the first validation example.
example = validation_data[0]
chars, mask, _ = example

# Networks only process batches. Create a batch of size one.
chars_batch, mask_batch = chars.unsqueeze(0), mask.unsqueeze(0)

# Copy batch to the GPU.
chars_batch, mask_batch = chars_batch.cuda(), mask_batch.cuda()

# Run the untrained network.
logits = untrained_baseline(chars_batch, mask_batch)

# Convert to a distribution.
output_distribution = F.softmax(logits, dim=1).squeeze().tolist()

# Inspect the distribution, which should be uniform.
list(zip(all_tags, output_distribution))


# Finally, time to write some code!
# 
# In the cell below, define a predictor for a network by following the instructions in the comments. The predictor takes a list of words (strings) and returns a list of part-of-speech tags (also strings).
# 
# For this assignment, we've provided more fine-grained instructions as comments in the code template.  You are free to explore methods and architectures other than the ones we specified in the comments, but we highly recommend starting with them, as they will help you reach the required accuracies and give lots of best practices to use in later projects.

# In[ ]:


def predict_using(network):
    def predictor(raw_data):
        """Return a list of part-of-speech tags as strings, one for each word.

        raw_data - a list of (word, tag) pairs.
        """

        with torch.no_grad(): # turns off automatic differentiation, which isn't required but helps save memory

            # YOUR CODE HERE
            # * put `network` into evaluation mode (turning off dropout) using `.eval()`
            #   then back into train mode at the end of the function with `.train()`
            #   this is easy to forget and could lead to lower accuracy without warning
            #   see https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.train for more info
            # * use `make_matrices` to get a preprocessed dataset from `raw_data`
            # * make a DataLoader to iterate over the preprocessed dataset from `make_matrices`, but don't use shuffling or your outputs will be in the wrong order
            # * iterate with the data loader (there will be a pos_labels vector, but don't use it - we want to be able to use our model on new inputs where we don't know the answer)
            #  * run `network` to get outputs
            #  * get the id of the predicted part of speeches with an argmax operation
            #  * convert the predictions to strings using `all_tags`
            # * return your predictions


    
    return predictor

# The predictions of an untrained model should be arbitrary.
print_sample_predictions(predict_using(untrained_baseline), validation_data_raw)


# 
# Fill in the training function for the neural network below. This function should train any network.  
# 
# Then, you'll have all the parts needed to train and evaluate the baseline network.  You should get the same accuracy as the all-noun baseline.  Make sure your train function prints validation scores so that you see score outputs here.

# In[ ]:


import tqdm

def train(network, n_epochs=25):
    # YOUR CODE HERE
    # * use `make_matrices` to get a preprocessed dataset from `train_data_raw`
    # * make a DataLoader from torch.utils.data to iterate over your dataset
    #   it can handle batching and shuffling of the data, you just need to pass it the `batch_size` and `shuffle` parameters
    # * move `network` to GPU using `.cuda()`
    # * make an optimizer from torch.optim with your network parameters
    #   `Adam` with its default hyperparameters often works pretty well without any tuning
    #   later you can explore other optimizers, as well as learning rate schedules
    
    predictor = predict_using(network)

    for epoch in range(n_epochs):
        print('Epoch', epoch)
        for batch in tqdm.tqdm_notebook(data_loader, leave=False):
            chars_batch, mask_batch, pos_batch = batch
            assert network.training, 'make sure your network is in train mode with `.train()`'

            # YOUR CODE HERE
            # * call zero_grad on your optimizer
            #   warning: this is easy to forget and you won't get an error if you do - you might just get lower accuracies
            # * move the batch inputs to GPU
            # * run your network
            # * compute a loss; you can use `F.cross_entropy`, which combines a softmax operation with
            #   a cross-entropy loss operation for multi-class classification
            # * call `.backward()` on your loss and `.step()` on your optimizer



        validation_score = evaluate(predictor, validation_data_raw)
        print('Validation score:', validation_score)

        # YOUR CODE HERE
        # * if the validation score is better than your previous best score, save the model
        #   use `network.state_dict()` and `torch.save` (https://pytorch.org/docs/stable/notes/serialization.html)
        #   this gives us a form of early stopping in case the model starts overfitting



    # YOUR CODE HERE
    # * load the best model from the file where you saved it using `torch.load` and `network.load_state_dict`
    #   and return it



trained_baseline_network = train(BaselineNetwork(len(all_tags)), 2)
print_sample_predictions(predict_using(trained_baseline_network), 
                         validation_data_raw)


# It's time to actually define a non-trivial neural network.  We'll start with a pretty simple model that takes embeddings of the characters of a word, pools them, and runs a feedforward network.  Fill in your code for `PoolingNetwork` below.  A correct implementation should get a validation score over 66%.

# In[ ]:


class PoolingNetwork(nn.Module):
    def __init__(self, n_outputs): # pass whatever arguments you need
        super().__init__() # you will get an error if you don't call the parent class __init__

        # YOUR CODE HERE
        # create Modules from torch.nn (imported as nn)
        # here you will need nn.Embedding and two nn.Linear
        # you may find it easier to start with the `forward` method and as you need components come back to place them here



    def forward(self, chars, mask): # the main method that runs this module
        # for this network, `chars` should be an int64 tensor of character ids with size (batch, n_chars)
        #   note that sometimes PyTorch puts sequence dimensions before the batch, so you will need to make sure you know which you are using
        # `mask` is a float32 tensor of size (batch, n_chars) that is 1.0 if the character at that position in `chars` is valid (else 0.0)
        # the function returns a float32 tensor of size (batch, n_pos)
        
        # we recommend that you return pre-activation values from modules (e.g. the values before running softmax or sigmoid)
        # pytorch has loss functions that combine the softmax/sigmoid operation into the loss operation for more numerical stability

        # YOUR CODE HERE
        # Your code should do the following:
        # 1) get character embeddings
        # 2) multiply embeddings by `mask` (you will need to use `view` or `unsqueeze` to make the broadcasting work correctly;
        #    see https://pytorch.org/docs/stable/notes/broadcasting.html)
        # 3) pool over the characters of each word with the Tensor `mean` function
        # 4) run a linear layer
        # 5) apply an activation (ReLU is a decent default choice; look in torch.nn.functional, which is imported as F)
        # 6) run dropout; you can either make a nn.Dropout module in __init__ or use F.dropout, but if you use F.dropout, be sure to pass training=self.training to
        #    make sure dropout gets turned off during evaluation
        # 7) run your second linear layer and return the output



trained_pooling_network = train(PoolingNetwork(len(all_tags)))
pooling_predictor = predict_using(trained_pooling_network)


# And look at some outputs.

# In[ ]:


print_sample_predictions(pooling_predictor, validation_data_raw)


# For this next part, we'll give you a little more freedom to experiment.  Think about what types of information could be useful for predicting parts of speech.  Think about what the pooling model is missing.  Implement an improved model that reaches a validation score above 75%.
# 
# One way to reach the required accuracy is to operate over character n-grams before pooling.
# There are several ways to implement this, but if you need help, you can use the following steps between the creation of embeddings and the mask/pool operations to process bigrams:
# 1. create two slices of the embedding tensor, one with the first character cut off and one with the last cut off
# 2. concatenate the two sliced tensors along the embedding dimension with `torch.cat`
# 3. run a linear layer with activation on the concatenated embeddings
# 4. cut off the first character of the mask tensor

# In[ ]:


class ImprovedNetwork(nn.Module):
    def __init__(self, n_outputs): # pass whatever arguments you need
        super().__init__()

        # YOUR CODE HERE
        # create Modules from torch.nn (imported as nn)



    def forward(self, chars, mask):
        # for this network, `chars` should be an int64 tensor of character ids with size (batch, n_chars)
        # `mask` is a float32 tensor of size (batch, n_chars) that is 1.0 if the character at that position in `chars` is valid (else 0.0)
        # the function returns a float32 tensor of size (batch, n_pos)

        # YOUR CODE HERE


trained_improved_network = train(ImprovedNetwork(len(all_tags)))
improved_predictor = predict_using(trained_improved_network)


# We can also get a feel for what our model learned by providing some of our own inputs that aren't real words (yet).

# In[ ]:


print_sample_predictions(improved_predictor, validation_data_raw)

print_sample_predictions(improved_predictor, [['kleining','X'], ['deneroful','X']])


# Finally, you need to run your model on the test set and save the outputs.  You'll turn in your predictions for us to grade.

# In[ ]:


def save_predictions(predictions, filename):
    """Save predictions to a file.
    
    predictions is a list of strings.
    """
    with open(filename, 'w') as f:
        for pred in predictions:
            f.write(pred)
            f.write('\n')

print('test score pooling:', evaluate(pooling_predictor, test_data_raw))
print('test score improved:', evaluate(improved_predictor, test_data_raw))

test_predictions = pooling_predictor(test_data_raw)
save_predictions(test_predictions, 'predicted_test_outputs_pooling.txt')
test_predictions = improved_predictor(test_data_raw)
save_predictions(test_predictions, 'predicted_test_outputs_improved.txt')

# Check that your test set looks like we expect it to
import hashlib
m = hashlib.md5()
m.update(str(test_data_raw).encode('utf-8'))
assert m.digest() == b'*N\xf6\xbe\xed\xde\xe8q)\xb9GG\xa6\x15UI'


# ### Gradescope
# 
# We will use Gradescope for assignment submission.  You should have received an email with instructions to get added to Gradescope.  If not, please make a private post on Piazza.
# 
# You will submit this notebook and the required output files that we specify.  For this project, your submission will contain:
# * hw1a.ipynb
# * predicted_test_outputs_pooling.txt
# * predicted_test_outputs_improved.txt
# 
# You can upload files individually or as part of a zip file, but if using a zip file be sure you are zipping the files directly and not a folder that contains them.
#  
# To download this notebook, go to `File->Download .ipynb`.  Please rename the file to match the name in our file list.  You can download other outputs, like `predicted_test_output_improved.txt` by clicking the > arrow near the top left and finding it under `Files`.
# 
# When submitting your ipython notebooks, make sure everything runs correctly if the cells are executed in order starting from a fresh session.  Note that just because a cell runs in your current session doesn't mean it doesn't rely on code that you have already changed or deleted.  If the code doesn't take too long to run, we recommend re-running everything with `Runtime->Restart and run all...`.
# 
# When you upload your submission to the Gradescope assignment, you should get immediate feedback that confirms your submission was processed correctly.  Be sure to check this, as an incorrectly formatted submission could cause the autograder to fail.  For this project, you should be able to see your test set accuracies and a confirmation that all required files were found, but you will not be able to see your score until later.  Most assignments will be graded primarily on your test set accuracies, but we may also use other factors to grade.
# 
# Note that Gradesope will allow you to submit multiple times before the deadline, and we will use the latest submission for grading.
