#################################################################################################################################################
#Authorship of each file :- 
#		Source code          - 			Shireen
# 		BASELINES            - 			Radhika
#		human annotated.csv  - 	        Radhika
#		ORIGINS              -			Radhika
#		INSTALLS             - 			Shireen
#		  
#1. Program Name: "Topic Modelling on 6 US Airlines" by "Shireen"
#   Problem: Business owners would want to know what their customers think about their brand and services. Topic modelling would help them identify
#			 areas they would require improvement. Net promoter score(NPS) is calculated for each topic for each airline. 
#
#2. Program Run:- The program is run from the command line as follows:
#    python FinalProject.py tweets.csv LDAModelPP.txt topics.txt results.csv  > nps.txt
#	   FinalProject.py        -     Python code to run topic modelling
#      tweets.csv	          - 	Main csv file consisting of tweets 
#	   LDAModelPP.txt         -     To load the saved LDA model
#	   topics.txt             -     Results of generated topics
#	   results.csv            -     Results when topics are reapplied back to the tweets
#      nps.txt                -     NPS score for each topic for each airline
#   		
#3. [i] The tweets from the csv file are taken into a list. Data cleaning is performed on the tweets.		 
#  [ii] Punctuations are removed. Texts converted into lower case and then word tokenized. Stop words and numbers are removed. Lemmatization is 
#		used to bring words into its canonical form. Stop words not removed by nltk stop list such as 'could', 'would' etc are removed manually.
#		Top 100 words by frequencies are searched and words found redundant such as 'flight', 'flightled', 'amp' are removed.   
# [iii] Using phraser module of gensim package bigrams are created of words that appear often together. A dictionary is created where each token
#		(word) is identified with a unique number along with its counts.  
#  [iv]	LDA model is applied to number of topics(k) from 1 to 20 and coherence scores are calculated for each k. A graph is plotted to see the 
#		trend of increase. A value is selected such that there is a dip after sudden increase (see paper for more details). A value of k=7 is 
#		selected to be the optimal model, which is confirmed by visualizations using pyLDAvis. The topics for k=7 are written into topics.txt.
#  	[v] The model is then applied back to the corpus to label each tweet with a highest probable topic number. These results are saved to 
#		results.csv.
#  [vi] Data is divided based on airline. For each airline, topic number and sentiments are taken into a dictionary as key and values.
# [vii] NPS scores are calculated as ((number of positives - number of negatives)/total number of tweets per topic per airline) and outputted
#		into nps.txt.
#
# Note:-
#
# Human annotation is used as baseline model as this is an unsupervised learning approach. 10 tweets per airline were analysed and scored to  
# compute accuracy. 50 tweets were found to be accurately classified by the model out of the 60 tweets.
# 								ACCURACY = 50/60 = 83.33% 

import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
import string, re
from nltk.stem import WordNetLemmatizer
import gensim
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel
#import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim
import argparse

# To take files from command line.
parser = argparse.ArgumentParser()
parser.add_argument("file",type = str, nargs='+',help = "Enter 3 file names: .csv, .txt, .csv") 
args = parser.parse_args()

# To read main file consisting of tweets
df = pd.read_csv(args.file[0], delimiter=',')
text=[]
text = list(df['text'])
plane=list(df['airline'])

# DATA CLEANING
text = [''.join(c for c in s if c not in string.punctuation) for s in text]               # To remove punctuation
text = [i.lower() for i in text]                                                          # To convert into lower case
stop = set(stopwords.words('english'))
text = [word_tokenize(i) for i in text]   
tweets = []
text1 = []
for i in text:                                                                            
    i = [w for w in i if not w in stop]                                                   # To remove stop words
    i = [w for w in i if not re.search(r'^-?[0-9]+(.[0-9]+)?$', w)]                       # To remove numbers   
    text1.append(i)
wordnet_lemmatizer = WordNetLemmatizer()
lemmatized_token=[]
for sent in text1:                                                                        # Lemmatization to convert tokens to canonical form
    tweets = []
    for token in sent:
        token = wordnet_lemmatizer.lemmatize(token)
        token = wordnet_lemmatizer.lemmatize(token,pos='v')
        tweets.append(token)
    lemmatized_token.append(tweets)
	
##########################################################################
# COMMENTED INTENTIONALLY                                                #
# Code to find top 100 words as per frequency to check for unwanted and  #
# redundant words. Added stop words based on this analysis.              #
########################################################################## 
#from collections import Counter
#count=[]
#for i in lemmatized_token:
#    for j in i:
#        count.append(j)
#count
#count=Counter(word for word in count)
#count.most_common(100)

# To remove added stop words as analysed from above commented code
added_stop_words = ['much','two','youve','ever','since','aa','jfk','dfw','americanairlines','didnt','thats','still','ive','really','virginamerica','usairways','americanair','southwestair','jetblue','airline','\'','\"','plane','flight','flightled','wo','hows', 'unite', 'u', 'get','im','could','would','make','cant','amp','dont','number','one','need','w','theyd','take','even']
lemmatized_token = [[c for c in s if c not in added_stop_words] for s in lemmatized_token]

# MODELLING
bigram = gensim.models.Phrases(lemmatized_token)  
bigram_mod = gensim.models.phrases.Phraser(bigram)
lemmatized_token = [bigram_mod[line] for line in lemmatized_token]  				# bigrams for words appearing together often created  
dictionary = Dictionary(lemmatized_token)                                           # creating dictionary to identify mapping between tokens and their integer ids											
dictionary.filter_extremes(no_below=2)                                              # keep tokens which are contained in atleast in 2 tweets 
corpus = [dictionary.doc2bow(text) for text in lemmatized_token]					# gives number to each unique token along with its counts in corpus

##########################################################################
# COMMENTED INTENTIONALLY                                                #
# Code to choose optimal k. Loop to calculate coherence scores of number #
# of topics from 1 to 20.                                                #
##########################################################################
#limit=21
#start=1 
#step=1
#coherence_values = []
#model_list = []
#for num_topics in range(start, limit, step):
#    model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
#    model_list.append(model)
#    coherencemodel = CoherenceModel(model=model, texts=lemmatized_token, dictionary=dictionary, coherence='c_v')
#    coherence_values.append(coherencemodel.get_coherence())
#coherence_values

##########################################################################
# COMMENTED INTENTIONALLY                                                #
# Graph to plot coherence scores vs number of topics.                    #
# Visualization for number of topics equal to 7							 #
##########################################################################
#limit=21; start=1; step=1;
#x = range(start, limit, step)
#plt.plot(x, coherence_values)
#plt.xlabel("Num Topics")
#plt.ylabel("Coherence score")
#plt.legend(("coherence_values"), loc='best')
#plt.show()																			# graph to plot coherence scores vs number of topics
#pyLDAvis.enable_notebook()
#vis = pyLDAvis.gensim.prepare(model_list[6], corpus, dictionary)                   # visualization for number of topics equal to 7
#vis

#model_list[6].save("C:/Users/shailaja/Desktop/Shireen/NLP/Final Project/LDAModelPP.txt")         # to chosen lda model

# Topic Names :
# Topic 0 - call_center 
# Topic 1 - waiting
# Topic 2 - cancelling
# Topic 3 - delays
# Topic 4 - customer_service
# Topic 5 - travel
# Topic 6 - thanks
a = args.file[1]
lda_model = LdaModel.load(a)														# to load lda model
#print(lda_model.show_topics())

# To write topics obtained to file
topics = open(args.file[2],'w')
topics.write(str(lda_model.show_topics()))

# PROCEDURE TO APPLY TOPICS BACK TO THE DATASET
lda_new=lda_model[corpus]															#applying lda model back to corpus to get probailities
i=0
review_lda = []
while i<len(lda_new): 
    review_lda.append(sorted(lda_new[i], key=lambda k:-k[1])[:1] )					#sort the topics according to highest probability for a particular tweet		
    i = i+1
df_scored = pd.DataFrame(review_lda)                                                #convert above sorted data into a DataFrame
df_scored.columns = ['topic_number']
df_scored['topic_number'] = df_scored['topic_number'].astype(str)                   #Convert the datatype of topic number to string
df_scored["topic_number"]= df_scored['topic_number'].str.replace(r'(', '').str.strip()   
df_scored["topic_number"]= df_scored['topic_number'].str.replace(r')', '').str.strip()
df_scored['topic_number'] = df_scored['topic_number'].str.split(',').str[0]         #first topic of sorted dataframe retained and others removed   
df_text = pd.read_csv(args.file[0], delimiter=',')
result=df_text.join(df_scored)
result.to_csv(args.file[3])

# NET PROMOTER SCORE
sentiment = list(result['airline_sentiment'])										# list of sentiments
airline= list(result['airline'])													# list of airlines
topic = list(result['topic_number'])												# list of topic numbers
# Dictionary for each airline to store topic number as key and sentiment as value
American = {}
Virgin = {}
United = {}
Delta = {}
SW = {}
US = {}
for i, j, k in zip(airline,topic,sentiment):
    if i == 'American':
        American.setdefault(j, []).append(k)
    elif i == 'Virgin America':
        Virgin.setdefault(j, []).append(k)
    elif i == 'United':
        United.setdefault(j, []).append(k)
    elif i == 'Delta':
        Delta.setdefault(j, []).append(k)
    elif i == 'Southwest':
        SW.setdefault(j, []).append(k)
    elif i == 'US Airways':
        US.setdefault(j, []).append(k)
     
# Display nps score for each topic for each airline	 
nps=[]
for key,value in American.items():
    p=0
    n=0
    for i in range(len(value)):
        if value[i] == 'positive':
            p+=1
        elif value[i] == 'negative':
            n+=1
    nps.append((p-n)*100/len(value))
    print("American Airlines - NPS of topic "+key+" is:"+str((p-n)*100/len(value)))			# formula to calculate nps			
for key,value in Virgin.items():
    p=0
    n=0
    for i in range(len(value)):
        if value[i] == 'positive':
            p+=1
        elif value[i] == 'negative':
            n+=1
    nps.append((p-n)*100/len(value))
    print("Virgin Airlines - NPS of topic "+key+" is:"+str((p-n)*100/len(value))) 			# formula to calculate nps
for key,value in United.items():
    p=0
    n=0
    for i in range(len(value)):
        if value[i] == 'positive':
            p+=1
        elif value[i] == 'negative':
            n+=1
    nps.append((p-n)*100/len(value))
    print("United Airlines - NPS of topic "+key+" is:"+str((p-n)*100/len(value)))			# formula to calculate nps
for key,value in Delta.items():
    p=0
    n=0
    for i in range(len(value)):
        if value[i] == 'positive':
            p+=1
        elif value[i] == 'negative':
            n+=1
    nps.append((p-n)*100/len(value))
    print("Delta Airlines - NPS of topic "+key+" is:"+str((p-n)*100/len(value)))			# formula to calculate nps
for key,value in SW.items():
    p=0
    n=0
    for i in range(len(value)):
        if value[i] == 'positive':
            p+=1
        elif value[i] == 'negative':
            n+=1
    nps.append((p-n)*100/len(value))
    print("Southwest Airlines - NPS of topic "+key+" is:"+str((p-n)*100/len(value)))		# formula to calculate nps
for key,value in US.items():
    p=0
    n=0
    for i in range(len(value)):
        if value[i] == 'positive':
            p+=1
        elif value[i] == 'negative':
            n+=1
    nps.append((p-n)*100/len(value))
    print("US Airways - NPS of topic "+key+" is:"+str((p-n)*100/len(value)))				# formula to calculate nps