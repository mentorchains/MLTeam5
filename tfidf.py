#%%
import pickle 
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize as st
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk.corpus import wordnet
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics
from sklearn.feature_selection import chi2
from IPython.display import display
from sklearn.model_selection import train_test_split

#%%
#using webscraped data in the form of nested lists stored as pickle files
with open("Ishan_home_assist_text_submissions11.pickle", "rb") as input_file:
    file1 = pickle.load(input_file)
with open("Ishan_home_assist_text_submissions11.pickle", "rb") as input_file:
      home_assist = pickle.load(input_file)
with open("Badhri-Hopscotch_text_submissions.pickle", "rb") as input_file:
      hopscotch = pickle.load(input_file)
with open("saiparsa_Choice_Community_text.pickle", "rb") as input_file:
      choice_community = pickle.load(input_file)
with open("shreya_codecademy_text_submissions.pickle", "rb") as input_file:
      code_academy = pickle.load(input_file)

#%%

#preprocessing the posts from the different forms

choice_community10 = [[x[0], ' '.join(x[1])] for x in choice_community]

def preprocessing(forum_data):

    forum_data = [[x[0], re.sub('\n',' ',x[1])] for x in forum_data]
    #forum_data = [[x[0], re.sub(r'\W', ' ',x[1])] for x in forum_data]
    forum_data = [[x[0], re.sub(r'\s+[a-zA-Z]\s+', ' ',x[1])] for x in forum_data]
    forum_data = [[x[0], re.sub(r'\^[a-zA-Z]\s+',' ',x[1])] for x in forum_data]
    forum_data = [[x[0], re.sub(r'\s+',' ',x[1])] for x in forum_data]
    forum_data = [[x[0], re.sub( r'^b\s+',' ',x[1])] for x in forum_data]
    forum_data = [[x[0], x[1].lower()] for x in forum_data]
    forum_data = [[x[0], x[1].split()] for x in forum_data]
    stemmer = WordNetLemmatizer()
    forum_data = [[x[0], [stemmer.lemmatize(word) for word in x[1]]] for x in forum_data]
    forum_data = [[x[0], ' '.join(x[1])] for x in forum_data]
    return forum_data

home_assist1 = preprocessing(home_assist)
choice_community1 = preprocessing(choice_community10)
hopscotch1 = preprocessing(hopscotch)
code_academy1 =preprocessing(code_academy)
#%%
#convert paragraphs into sentences
nltk.download('punkt')
home_assist1 = [[x[0], st(x[1])] for x in home_assist1]
choice_community1 = [[x[0], st(x[1])] for x in choice_community1]
hopscotch1 = [[x[0], st(x[1])] for x in hopscotch1]
code_academy1 = [[x[0], st(x[1])] for x in code_academy1]

#%%
#convert list data into dataframes
def convert_to_df(df):
  df1 = pd.DataFrame(df)
  df1.columns = ['Links' , 'Posts']
  df1['Posts_string'] = [','.join(map(str, l)) for l in df1['Posts']]
  del df1['Links']
  return df1

home_assist2 = convert_to_df(home_assist1)
home_assist2.insert(0, 'Forum_Name', 'Home_Assist')

choice_community3 = convert_to_df(choice_community1)
choice_community3.insert(0, 'Forum_Name', 'Choice Community')

hopscotch2 = convert_to_df(hopscotch1)
hopscotch2.insert(0, 'Forum_Name', 'Hopscotch')

code_academy2 = convert_to_df(code_academy1)
code_academy2.insert(0, 'Forum_Name', 'Code Academy')

forums = [home_assist2,code_academy2,hopscotch2,choice_community3]
result_df = pd.concat(forums)

#%%
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,ngram_range=(1, 2), stop_words='english')
result_df['forum_id'] = result_df['Forum_Name'].factorize()[0]
result_df1 = result_df.sample(frac=1)

#%%
features = tfidf.fit_transform(result_df1.Posts_string)
labels = result_df1.Forum_Name
#%%
X = result_df1['Posts_string'] 
y = result_df1['Forum_Name']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)

#%%
X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(features, labels, result_df1.index, test_size=0.25, random_state=1)
model2 = MultinomialNB()
model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)
#%%
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))