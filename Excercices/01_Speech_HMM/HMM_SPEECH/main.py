
# coding: utf-8

# # Installation
#
# The following libraries are required for this TP:
# - librosa
# - numpy
# - sklearn
# - warnings
# - HMMlearn
#
#
# ### HMM learn
# **hmmlearn** is a set of algorithms for unsupervised learning and inference of **Hidden Markov Models**.
# If you have python 3.6.* or 3.7.* you can simply install hmmlearn with the command:
# ```
# pip install hmmlearn
# ```
#

# In[1]:


import sys

print(sys.version_info[1])

if sys.version_info[1] < 6:
    print("Attention version < 6")
    # Fix issue in paramz
    import re

    re._pattern_type = re.Pattern# Imports



# In[2]:


from hmmlearn import hmm


# In[3]:


import librosa
from librosa import load
import numpy as np
import sklearn
import warnings
warnings.filterwarnings('ignore')


# # Constants

# In[4]:


SR = 16000 # Audio sampling rate
HOP_LENGTH = 128 # Size of a frame
NB_COEFFICIENT = 12 # For MFCC (8ms at 16kHz)


# # Read audio files

# In[5]:


# The audio files are in a 'wave' folder
# Train set
train_1 = [load("wave/1_1.wav", SR)[0], load("wave/1_2.wav", SR)[0], load("wave/1_3.wav", SR)[0]] # "un"
train_2 = [load("wave/2_1.wav", SR)[0], load("wave/2_2.wav", SR)[0], load("wave/2_3.wav", SR)[0]] # "deux"
train_3 = [load("wave/3_1.wav", SR)[0], load("wave/3_2.wav", SR)[0], load("wave/3_3.wav", SR)[0]] # ...
train_4 = [load("wave/4_1.wav", SR)[0], load("wave/4_2.wav", SR)[0], load("wave/4_3.wav", SR)[0]]
train_5 = [load("wave/5_1.wav", SR)[0], load("wave/5_2.wav", SR)[0], load("wave/5_3.wav", SR)[0]]

# Test set
test_1 = [load("wave/1t.wav", SR)[0]] # "un"
test_2 = [load("wave/2t.wav", SR)[0]] # "deux"
test_3 = [load("wave/3t.wav", SR)[0]] # ...
test_4 = [load("wave/4t.wav", SR)[0]]
test_5 = [load("wave/5t.wav", SR)[0]]
test_p = [load("wave/peu.wav", SR)[0]]


# # Feature extraction

# In[6]:


def mfcc(y):
    """
    Apply Mel-frequency cepstral coefficients (MFCCs) on a audio time serie.
    Hypothesis:
    - Sampling rate is at costant sampling rate (SR)
    """
    mfccs = librosa.feature.mfcc(y, SR, n_mfcc=NB_COEFFICIENT, hop_length=HOP_LENGTH)

    mfccs = sklearn.preprocessing.scale(mfccs, axis=1) # Scale to unit variance and zero mean
    return mfccs.transpose() # To have frames on the 0 axis


# In[7]:


train_1_c = list(map(mfcc, train_1)) # cepstr
train_2_c = list(map(mfcc, train_2))
train_3_c = list(map(mfcc, train_3))
train_4_c = list(map(mfcc, train_4))
train_5_c = list(map(mfcc, train_5))

test_1_c = list(map(mfcc, test_1))
test_2_c = list(map(mfcc, test_2))
test_3_c = list(map(mfcc, test_3))
test_4_c = list(map(mfcc, test_4))
test_5_c = list(map(mfcc, test_5))
test_p_c = list(map(mfcc, test_p))


# # Durations and number of acoustic vectors

# In[35]:


def print_duration_nb_accoustic_models(time_series, cepstr):
    # TODO

    i = 0
    while i < 3:
        print("Duration [ms]: {}    -   Nb. accoustic vectors: {}".format(((len(time_series[i]) / SR) * 1000), (cepstr[i].shape[0]) ))
        i += 1

    # example of possible result:
    # Duration [ms]: 962.3125
    # Nb. accoustic vectors: 121


# In[36]:


print("Train 1:")
print_duration_nb_accoustic_models(train_1, train_1_c)
print("Train 2:")
print_duration_nb_accoustic_models(train_2, train_2_c)
print("Train 3:")
print_duration_nb_accoustic_models(train_3, train_3_c)
print("Train 4:")
print_duration_nb_accoustic_models(train_4, train_4_c)
print("Train 5:")
print_duration_nb_accoustic_models(train_5, train_5_c)


# # Train a model for each class

# In[10]:


def concatenate_cepstrums(dataset):
    """
    hmmlearning can ingest a long serie containing multiple series
    you need to pass the serie and length of each
    """
    X = np.concatenate(dataset)
    lengths = [c.shape[0] for c in dataset]
    return X, lengths


# ## "un"

# In[13]:
from hmmlearn.base import ConvergenceMonitor

class ThresholdMonitor(ConvergenceMonitor):
    @property
    def converged(self):
         return (self.iter == self.n_iter or
                 self.history[-1] >= self.tol)

N = 5
X, lengths = concatenate_cepstrums(train_1_c)
model_1 = hmm.GaussianHMM(n_components=N, verbose=True)
print("model_1 : ")

model_1.fit(X, lengths)
# ## "deux"

# In[14]:


N = 6
X, lengths = concatenate_cepstrums(train_2_c)
model_2 = hmm.GaussianHMM(n_components=N, verbose=True)
print("model_2 : ")
model_2.fit(X, lengths)


# ## "trois"

# In[15]:


N = 8
X, lengths = concatenate_cepstrums(train_3_c)
model_3 = hmm.GaussianHMM(n_components=N, verbose=True)
print("model_3 : ")
model_3.fit(X, lengths)


# ## "quattre"

# In[16]:


N = 8
X, lengths = concatenate_cepstrums(train_4_c)

model_4 = hmm.GaussianHMM(n_components=N, verbose=True)
print("model_4 : ")
model_4.fit(X, lengths)


# ## "cinq"

# In[17]:


N = 7
X, lengths = concatenate_cepstrums(train_5_c)
model_5 = hmm.GaussianHMM(n_components=N, verbose=True)
print("model_5 : ")
model_5.fit(X, lengths)


# In[13]:


N = 5
X, lengths = concatenate_cepstrums(train_1_c)
model_1 = hmm.GaussianHMM(n_components=N, verbose=True)
print("model_1 : ")
model_1.fit(X, lengths)


# # Test unseen data

# In[18]:


def test(dataset, models):
    X, lengths = concatenate_cepstrums(dataset)
    scores = [m.score(X, lengths) for m in models]
    print(scores)
    print(f"Best model: {np.argmax(scores) + 1}")


# ## "un"

# In[19]:

print("\n-------- TESTS DES MODELES SIMPLE----------- \n")

test(test_1_c, [model_1, model_2, model_3, model_4, model_5])


# ## "deux"

# In[20]:


test(test_2_c, [model_1, model_2, model_3, model_4, model_5])


# ## "trois"

# In[21]:


test(test_3_c, [model_1, model_2, model_3, model_4, model_5])


# ## "quattre"

# In[22]:


test(test_4_c, [model_1, model_2, model_3, model_4, model_5])


# ## "cinq"

# In[23]:


test(test_5_c, [model_1, model_2, model_3, model_4, model_5])


# ## "peu"

# In[24]:


# TODO: tester avec le mot peu
test(test_p_c, [model_1, model_2, model_3, model_4, model_5])

# # Train a model for each class using LR Transition Probabilities

# ## "un"

# In[55]:


N = 5
X, lengths = concatenate_cepstrums(train_1_c)
model_1_lr = hmm.GaussianHMM(n_components=N, covariance_type="diag", init_params="cm", params="cmt", verbose=True)
print("model_1_lr : ")

model_1_lr.startprob_ = np.array(   [1.0, 0.0, 0.0, 0.0, 0.0])
model_1_lr.transmat_ = np.array([   [0.5, 0.5, 0.0, 0.0, 0.0],
                                    [0.0, 0.5, 0.5, 0.0, 0.0],
                                    [0.0, 0.0, 0.5, 0.5, 0.0],
                                    [0.0, 0.0, 0.0, 0.5, 0.5],
                                    [0.0, 0.0, 0.0, 0.0, 1.0]])
model_1_lr.fit(X, lengths)


# ## "deux"

# In[41]:


N = 6

X, lengths = concatenate_cepstrums(train_2_c)
model_2_lr = hmm.GaussianHMM(n_components=N, covariance_type="diag", init_params="cm", params="cmt", verbose=True)
print("model_2_lr : ")
model_2_lr.startprob_ = np.array(   [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
model_2_lr.transmat_ = np.array([   [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.5, 0.5, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
model_2_lr.fit(X, lengths)
# ## "trois"

# In[42]:


N = 8
X, lengths = concatenate_cepstrums(train_3_c)
model_3_lr = hmm.GaussianHMM(n_components=N, covariance_type="diag", init_params="cm", params="cmt", verbose=True)
print("model_3_lr : ")
model_3_lr.startprob_ = np.array(   [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
model_3_lr.transmat_ = np.array([   [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
model_3_lr.fit(X, lengths)


# ## "quattre"

# In[44]:


N = 8
X, lengths = concatenate_cepstrums(train_4_c)
model_4_lr = hmm.GaussianHMM(n_components=N, covariance_type="diag", init_params="cm", params="cmt", verbose=True)
print("model_4_lr : ")
model_4_lr.startprob_ = np.array(   [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
model_4_lr.transmat_ = np.array([   [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
model_4_lr.fit(X, lengths)


# ## "cinq"

# In[45]:


N = 7
X, lengths = concatenate_cepstrums(train_5_c)
model_5_lr = hmm.GaussianHMM(n_components=N, covariance_type="diag", init_params="cm", params="cmt", verbose=True)
print("model_5_lr : ")
model_5_lr.startprob_ = np.array(   [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
model_5_lr.transmat_ = np.array([   [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
model_5_lr.fit(X, lengths)


# # Test unseen data (LR topology)

# In[46]:


def test(dataset, models):
    X, lengths = concatenate_cepstrums(dataset)
    scores = [m.score(X, lengths) for m in models]
    print(scores)
    print(f"Best model: {np.argmax(scores) + 1}")


# ## "un"

# In[47]:

print("\n-------- TESTS DES MODELES LTR----------- \n")

test(test_1_c, [model_1_lr, model_2_lr, model_3_lr, model_4_lr, model_5_lr])


# ## "deux"

# In[48]:


test(test_2_c, [model_1_lr, model_2_lr, model_3_lr, model_4_lr, model_5_lr])


# ## "trois"

# In[49]:


test(test_3_c, [model_1_lr, model_2_lr, model_3_lr, model_4_lr, model_5_lr])


# ## "quattre"

# In[50]:


test(test_4_c, [model_1_lr, model_2_lr, model_3_lr, model_4_lr, model_5_lr])


# ## "cinq"

# In[51]:


test(test_5_c, [model_1_lr, model_2_lr, model_3_lr, model_4_lr, model_5_lr])


# ## "peu"

# In[52]:


test(test_p_c, [model_1_lr, model_2_lr, model_3_lr, model_4_lr, model_5_lr])


# # Test on colleague data

# In[53]:


coll_test_1 = list(map(mfcc, [load("wave/colleague/1t.wav", SR)[0]]))
coll_test_2 = list(map(mfcc, [load("wave/colleague/2t.wav", SR)[0]]))
coll_test_3 = list(map(mfcc, [load("wave/colleague/3t.wav", SR)[0]]))
coll_test_4 = list(map(mfcc, [load("wave/colleague/4t.wav", SR)[0]]))
coll_test_5 = list(map(mfcc, [load("wave/colleague/5t.wav", SR)[0]]))


# ## "un"

# In[54]:

print("\n-------- TESTS DES FICHIERS D'UN COLLEGUE (MODELES SIMPLE)----------- \n")

test(coll_test_1, [model_1, model_2, model_3, model_4, model_5])


# ## "deux"

# In[29]:


test(coll_test_2, [model_1, model_2, model_3, model_4, model_5])


# ## "trois"

# In[30]:


test(coll_test_3, [model_1, model_2, model_3, model_4, model_5])


# ## "quattre"

# In[31]:


test(coll_test_4, [model_1, model_2, model_3, model_4, model_5])


# ## "cinq"

# In[32]:


test(coll_test_5, [model_1, model_2, model_3, model_4, model_5])


print("\n-------- TESTS DES FICHIERS D'UN COLLEGUE (MODELES LTR)----------- \n")

test(coll_test_1, [model_1_lr, model_2_lr, model_3_lr, model_4_lr, model_5_lr])

# ## "deux"
test(coll_test_2, [model_1_lr, model_2_lr, model_3_lr, model_4_lr, model_5_lr])

# ## "trois"
test(coll_test_3, [model_1_lr, model_2_lr, model_3_lr, model_4_lr, model_5_lr])

# ## "quattre"
test(coll_test_4, [model_1_lr, model_2_lr, model_3_lr, model_4_lr, model_5_lr])

# ## "cinq"
test(coll_test_5, [model_1_lr, model_2_lr, model_3_lr, model_4_lr, model_5_lr])