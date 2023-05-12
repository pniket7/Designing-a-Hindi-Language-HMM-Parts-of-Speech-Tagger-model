#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re


# In[2]:


file_path = "/home/niket/Desktop/Hindi_Treebank.txt"


# In[3]:


def preprocess_conll(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        contents = f.read()
    
    # remove extra newline characters
    contents = re.sub('\n{2,}', '\n', contents)
    
    # remove spaces before and after tab characters
    contents = re.sub('\s+\t', '\t', contents)
    contents = re.sub('\t\s+', '\t', contents)
    
    # split contents into individual sentences
    sentences = contents.split('\n\n')
    
    # remove empty sentences
    sentences = [s for s in sentences if len(s) > 0]
    
    # split each sentence into individual tokens
    tokenized_sentences = [s.split('\n') for s in sentences]
    
    # split each token into its components (word, POS tag)
    split_tokens = [[t.split('\t') for t in s] for s in tokenized_sentences]
    
    # return preprocessed CoNLL data
    return split_tokens

# preprocess CoNLL file
preprocessed_data = preprocess_conll(file_path)


# In[4]:


#path to save preprocessed file
save_path = "/home/niket/Desktop/File.txt"


# In[5]:


import nltk
from nltk.tag import hmm


# In[6]:


# Load the dataset from the file
with open("/home/niket/Desktop/file.txt", "r", encoding="utf-8") as f:
    dataset = f.read()


# In[7]:


# Split the dataset into individual sentences
sentences = dataset.strip().split("\n\n")


# In[8]:


tagged_sentences = []
for sentence in sentences:
    tagged_tokens = []
    lines = sentence.strip().split("\n")
    for line in lines:
        # Skip the metadata line
        if line.startswith("#"):
            continue
        columns = line.strip().split("\t")
        tagged_tokens.append((columns[1], columns[4]))
    tagged_sentences.append(tagged_tokens)


# In[9]:


# Train a HMM tagger on the tagged sentences
hmm_tagger = nltk.HiddenMarkovModelTagger.train(tagged_sentences)


# In[10]:


# Split the dataset into training and testing sets
train_size = int(0.8 * len(tagged_sentences))
train_sents = tagged_sentences[:train_size]
test_sents = tagged_sentences[train_size:]


# In[11]:


# Evaluate the tagger on the testing set
accuracy = hmm_tagger.evaluate(test_sents)
print(f"Accuracy: {accuracy:.2%}")


# In[12]:


#Test the tagger on a sample sentence
sample_sentence = "वह रोज कड़ी मेहनत करता है"
tagged_words = hmm_tagger.tag(nltk.word_tokenize(sample_sentence))
print(tagged_words)


# In[ ]:




