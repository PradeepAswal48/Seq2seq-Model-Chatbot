#Biluding the Chatbot with Deep NLP

#importing libraries
#3
import numpy as np
import tensorflow as tf
import re 
import time

#Data preprocessing - PART 1
lines= open('movie_lines.txt',encoding='utf-8', errors = 'ignore').read().split('\n')
conversations= open('movie_conversations.txt',encoding='utf-8', errors = 'ignore').read().split('\n')

#creating dictionary which lines to its ID
id2line = {}
for line in lines:
    _line = line.split(" +++$+++ ")
    if(len(_line) == 5):
        id2line[_line[0]] = _line[4]
        
#creating list of all conversation
conversation_ids = []
for converstaion in conversations[:-1]:
    _conversation = converstaion.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conversation_ids.append(_conversation.split(","))

#getting seperately questions and answers
questions = []
answers = []
for conversation in conversation_ids:
    for i in range(len(conversation)-1):
        questions.append(id2line.get(conversation[i]))
        answers.append(id2line.get(conversation[i+1]))
        
#doing first cleaning of text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am",text)
    text = re.sub(r"he's", "he is",text)
    text = re.sub(r"she's", "she is",text)
    text = re.sub(r"that's", "that is",text)
    text = re.sub(r"what's", "what is",text)
    text = re.sub(r"where's", "where is",text)
    text = re.sub(r"\'ll", " will",text)
    text = re.sub(r"\'ve", " have",text)
    text = re.sub(r"\'re", " are",text)
    text = re.sub(r"\'d", " would",text)
    text = re.sub(r"won't", "will not",text)
    text = re.sub(r"can't", "cannot",text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "",text)
    return text

#cleaning the question
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

#cleaning the answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))    


#creating dictionary that map each word to its number of accurences
word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word]= 1
        else:
            word2count[word] += 1
            
for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word]= 1
        else:
            word2count[word] += 1        
#creating 2 dictionary, that maps each word in questiona and answer to unique integer
threshold = 20
questionswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        questionswords2int[word] = word_number
        word_number += 1

answerswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        answerswords2int[word] = word_number
        word_number += 1

#adding the last token to these 2 dictionaries
tokens = ['<PAD>','<EOS>','<OUT>', '<SOS>']
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1 

for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1 


#craeting inverese dictionary of the answerswords2int dictionary
answersints2word = {w_i:w for w, w_i in answerswords2int.items()}

#Adding EOS to every answers
for i in range(len(clean_answers)):
    clean_answers[i] += " <EOS>" 

#translating all questions and answers to integer form
#and replacing all words below threshold by <OUT> token
    
questions_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if(word not in questionswords2int ):
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)
    
answers_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if(word not in answerswords2int ):
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)

#sorting questions and answer by their length
sorted_cleaned_questions = []
sorted_cleaned_answers = []
for length in range(1,25+1):
    for i in enumerate(questions_into_int):
        if len(i[1]) == length  :
            sorted_cleaned_questions.append(questions_into_int[i[0]])
            sorted_cleaned_answers.append(answers_into_int[i[0]])
    
#PART2 - building sec2sec Model
#creating placeholder for the inputs and the targets
            
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None,None], name="input")
    targets = tf.placeholder(tf.int32, [None,None], name="target")
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return inputs, targets, lr, keep_prob

#preprocessing the targets
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size,1],word2int['<SOS >'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size,-1],[1,1])
    preprocessed_targets = tf.concat([left_side,right_side],1)
    return preprocessed_targets
    

#creating encoding RNN layer
def encoder_rnn_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    









































