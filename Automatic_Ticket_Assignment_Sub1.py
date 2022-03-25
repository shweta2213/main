#!/usr/bin/env python
# coding: utf-8

# #### Load the Data

# In[1]:


import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.cm as cm
import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#load the full set of the data
# Read excel file
df = pd.read_excel('input_data.xlsx')
# Shape and size of the dataframe
print("> Shape of input_data :",df.shape)
print("\n> Size of input_data :",df.size)


# In[3]:


#check if the data frame is properly loaded using the sample() method
df.sample(10)


# In[4]:


#check if the data frame is properly loaded using the head() method
df.head(10)


# In[5]:


#check if the data frame is properly loaded using the tail() method
df.tail(10)


# In[6]:


df.columns


# In[7]:


#check the feature/columns  using the info method
df.info()


# In[8]:


df['Assignment group'].unique()


# In[9]:


df['Assignment group'].nunique()


# NOTE : There are 74 Dependent/Target classes which are required to be predicted by our Model

# In[10]:


targetClassCnt=df['Assignment group'].value_counts()
targetClassCnt


# In[11]:


targetClassCnt.describe()


# #### Observation:
# - Appears the Target class distrubtion is extremely skewed
# - Large no of entries for GRP_0 (mounting to 3976) which account for ~50% of the data
# - there are groups with 1 entry also. We could merge all groups with small entries to a group to reduce the imbalance in the target. This may reduce the imbalance to some extent.

# In[12]:


#chceck for na values
df.isna().sum()


# NOTE : There are 8 NaN Values in 'Short description' Feature, and 1 NaN value in 'Description' Feature.

# In[13]:


## find nulls
df[df.isnull().any(axis=1)]


# In[14]:


# For Short Description
df[df['Short description'].isnull()]


# In[15]:


# For Description
df[df['Description'].isnull()]


# In[16]:


# NULL replacement
# replacing nan values in short description
   
df["Short description"].fillna(df["Description"], inplace = True)


# In[17]:


# replacing nan values in description
df["Description"].fillna(df["Short description"], inplace = True)


# In[18]:


# verify the replacement
df.isnull().sum()


# In[19]:


# verify the replacement
print((df.iloc[2604]))


# In[20]:


# verify the replacement
print((df.iloc[4395]))


# In[21]:


#identify duplicate rows
duplicateRows = df[df.duplicated()]


# In[22]:


#view duplicate rows
duplicateRows


# In[23]:


# drop duplicate rows
df.drop_duplicates()


# #### Data Cleaning and Preprocessing

# #### Now clean up the Data to address the following
# 
# - Convert each character in a sentence to lowercase character
# - Remove HTML Tags
# - Remove punctuations
# - Remove stopwords
# - Remove common words like com, hello
# - Remove NUmbers - (r'[0-9] , is used for replace, but there are some numbers which remain in the data.. that needs to be tested further
# - Stemming was causing invalid words, hence used a lemmatizer

# In[24]:


import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 

def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    rem = re.sub('_xd_', '', rem_num)
    rem1 = re.sub('_', '', rem)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem1)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    stem_words=[stemmer.stem(w) for w in filtered_words]
    lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    clean_words = " ".join(filtered_words)
    return clean_words


# In[25]:


df['Short description']=df['Short description'].map(lambda s:preprocess(s)) 


# In[26]:


df['Description']=df['Description'].map(lambda s:preprocess(s)) 


# In[27]:


df['Caller']=df['Caller'].map(lambda s:preprocess(s)) 


# In[28]:


df.head(10)


# In[29]:


df.tail(10)


# In[30]:


df.sample(10)


# #### Observations:
# Index 2865 has sentences in different language than English. So we need to perform Translation for such rows.

# In[31]:


get_ipython().system('pip install deep-translator')


# In[69]:


#translation = translator.translate("¾ç½ ¹äº ï¼œæ žä¹ˆè šåž")
#from deep_translator import GoogleTranslator
#translation = GoogleTranslator(source='auto', target='en').translate("maus funktioniert nicht mehr richtig bitte")
#translation


# In[121]:


from deep_translator import GoogleTranslator
df['Description'] = df['Description'].apply(lambda x: GoogleTranslator(source='auto', target='en').translate(str(x)))


# In[ ]:


df['Short description'] = df['Short description'].apply(lambda x: GoogleTranslator(source='auto', target='en').translate(str(x)))


# In[42]:


# verify the replacement
print((df.iloc[3983]))


# #### Visualize the Dsitribution of Records across Groups

# In[32]:


import string
import collections


# In[33]:


all_data = df['Description'].str.split(' ')
all_data.head()


# In[34]:


all_data_cleaned = []

for text in all_data:
    text = [x.strip(string.punctuation) for x in text]
    all_data_cleaned.append(text)

all_data_cleaned[0]

text_data = [" ".join(text) for text in all_data_cleaned]
final_text_data = " ".join(text_data)
final_text_data[:500]


# In[35]:


import matplotlib.pyplot as pplt
wordcloud_desc = WordCloud(background_color="white").generate(final_text_data)

pplt.figure(figsize = (20,20))
pplt.imshow(wordcloud_desc, interpolation='bilinear')
pplt.axis("off")
pplt.show()


# In[36]:


all_data1 = df['Short description'].str.split(' ')
all_data1.head()


# In[37]:


all_data_cleaned1 = []

for text in all_data1:
    text = [x.strip(string.punctuation) for x in text]
    all_data_cleaned1.append(text)

all_data_cleaned1[0]

text_data1 = [" ".join(text) for text in all_data_cleaned1]
final_text_data1 = " ".join(text_data1)
final_text_data1[:500]


# In[38]:


wordcloud_sdesc = WordCloud(background_color="white").generate(final_text_data1)

pplt.figure(figsize = (20,20))
pplt.imshow(wordcloud_sdesc, interpolation='bilinear')
pplt.axis("off")
pplt.show()


# In[39]:


all_data2 = df['Caller'].str.split(' ')
all_data2.head()


# In[40]:


all_data_cleaned2 = []

for text in all_data2:
    text = [x.strip(string.punctuation) for x in text]
    all_data_cleaned2.append(text)

all_data_cleaned2[0]

text_data2 = [" ".join(text) for text in all_data_cleaned2]
final_text_data2 = " ".join(text_data2)
final_text_data2[:500]


# In[41]:


wordcloud_caller = WordCloud(background_color="white").generate(final_text_data2)

pplt.figure(figsize = (20,20))
pplt.imshow(wordcloud_caller, interpolation='bilinear')
pplt.axis("off")
pplt.show()


# In[58]:


## create a column to mark records with GRP_0 and non GRP_0=>GRP_X
df['GRP_MOD'] = df['Assignment group'].apply(lambda x: 'GRP_X' if x != 'GRP_0' else x)


# In[59]:


stopwords = set(STOPWORDS)
## function to create Word Cloud
def show_wordcloud(data, title):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(15, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()


# In[60]:


## view word cloud for GRP_0
text_Str = df['Description'][df['GRP_MOD'].isin(["GRP_0"])].tolist()
show_wordcloud(text_Str, "GRP_0 WORD CLOUD")


# In[61]:


## GRP_X Visualization:
text_Str = df['Description'][df['GRP_MOD'].isin(["GRP_X"])].tolist()
show_wordcloud(text_Str, "GRP_X WORD CLOUD")


# In[43]:


import matplotlib.pyplot as plt
descending_order = df['Assignment group'].value_counts().sort_values(ascending=False).index
plt.subplots(figsize=(22,5))
 
ax=sns.countplot(x='Assignment group', data=df, color='royalblue',order=descending_order)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.show()


# #### Feature Generation using Bag of Words

# Bag-of-words model(BoW ) is the simplest way of extracting features from the text. BoW converts text into the matrix of occurrence of words within a document. This model concerns about whether given words occurred or not in the document.

# In[49]:


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1))
text_counts= cv.fit_transform(df['Description'])


# #### Split train and test set

# In[50]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    text_counts, df['Description'], test_size=0.3, random_state=1)


# #### Model Building and Evaluation

# In[51]:


from sklearn.naive_bayes import MultinomialNB
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))


# #### Feature Generation using TF-IDF

# #### TFIDF Vectorization
# - We now Use TFIDF Vectorization method to convert the Words in the description to number vectors.

# In[52]:


from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
text_tf= tf.fit_transform(df['Description'])


# #### Split train and test set (TF-IDF)

# In[54]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    text_tf, df['Description'], test_size=0.3, random_state=123)


# #### Model Building and Evaluation (TF-IDF)

# In[55]:


from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))


# #### Handling Data Imbalance

# In[ ]:





# In[ ]:





# In[ ]:


sample = df.groupby(['Assignment group'])
regroup=[]
for grp in df['Assignment group'].unique():
    if(sample.get_group(grp).shape[0]<10):
        regroup.append(grp)
print('Found {} groups which have under 10 samples'.format(len(regroup)))
df['Assignment group']=df['Assignment group'].apply(lambda x : 'misc_grp' if x in regroup  else x)

# Unique Groups check 
df['Assignment group'].unique()


# In[ ]:


targetClassCnt=df['Assignment group'].value_counts()
targetClassCnt


# In[ ]:


df['Assignment group'].unique()


# In[ ]:


df['Assignment group'].nunique()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




