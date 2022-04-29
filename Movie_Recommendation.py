#!/usr/bin/env python
# coding: utf-8

# # MOVIE RECOMMENDATION SYSTEM

# In[89]:


# Importing Libraries :

import numpy as np 
import pandas as pd


# In[90]:


# Getting the data :

movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[91]:


movies.head(2)


# In[92]:


movies.shape


# In[93]:


credits.head(2)


# In[94]:


credits.shape


# In[95]:


# merge to dataframe on title #since title counts only once therefor it shows 23 column :

movies.merge(credits,on='title').shape 


# In[96]:


movies = movies.merge(credits,on='title')   # merging both the dataframe to the new variable 'movies'. 

movies.head(1)


# In[97]:


movies.shape


# In[98]:


movies.info()


# Here we can see there are lots of columns present. We will remove the unwanted columns. And keep the important columns necessary for analyses.

# In[99]:


# LIST OF NECESSARY COLUMNS 
# genres 
# id
# keywords
# originl_title change as title 
# overview
# cast 
# crew


# In[100]:


movies=movies[['movie_id','title','genres','keywords','overview','cast','crew']]

movies.head(1)          # the new dataframe which we will proceed with.


# In[101]:


# Getting the missing values :

movies.isnull().sum()


# In[102]:


# dropping the missing data :

movies.dropna(inplace=True)


# In[103]:


movies.isnull().sum()


# In[104]:


# To know the dulipcated data :

movies.duplicated().sum()


# There is no duplicate data present.

# In[105]:


# Preprocessing Columns :


# In[106]:


# Working on 'GENRES' column :

movies.iloc[0].genres


# In[107]:


# converting the output in a systematic format :

def convert(obj):
    List =[]
    for i in obj:
        List.append(i['name'])
        return List                               # This will not work proper because it takes integer value
                                                  # Therefore we import ast module


# In[108]:


import ast
def convert(obj):
    List =[]
    for i in ast.literal_eval(obj):
        List.append(i['name'])
    return List


# In[109]:


movies['genres'] =movies['genres'].apply(convert)


# In[110]:


movies.head(2)


# In[111]:


movies['keywords'] =movies['keywords'].apply(convert)        # Processing the 'keywords' columns


# In[112]:


movies.head(2)


# In[113]:


# Processing the 'CAST' columns :

movies['cast'][0]


# Here the cast contains lot of dictionary values . We will choose the 3 values and change the column accordingly .

# In[114]:


import ast
def convert3(obj):
    List =[]
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            List.append(i['name'])
            counter +=1
        else:
            break
    return List


# In[115]:


movies['cast'] =movies['cast'].apply(convert3) 


# In[116]:


movies.head(1)


# In[117]:


# GENRES, KEYWORD, CAST HAs been changed suxxessfully.


# In[118]:


# Preprocessing 'CREW' column :

# We have to extract the ditionary where the name of the should be the director : 

movies['crew'][0]


# In[119]:


def fetch_director(obj):
    List=[]
    counter=0
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            List.append(i['name'])
            break
    return List


# In[120]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[121]:


movies.head(1)


# In[122]:


movies['overview'][0]


# In[123]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[124]:


movies.head(1)

We have to remove the word space so as to avoid confusion . So that there should not be misleading in the recommendation system 
# In[126]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])

movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])

movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])

movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])


# In[127]:


movies.head(1)


# In[128]:


# Merging the columns :

movies['tags']= movies['overview']+ movies['genres']+ movies['keywords']+ movies['cast']+ movies['crew']


# In[129]:


new_df = movies[['movie_id','title','tags']]


# In[130]:


new_df.info()


# In[132]:


new_df.head(1)


# In[135]:


new_df['tags'].head(2)


# In[136]:


# The Tags columns is in form of LIST . So converting that in form of string :

new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))

new_df.head(1)


# In[138]:


new_df['tags'][0]


# In[139]:


# Converting in lower :

new_df['tags'] = new_df['tags'].apply(lambda  x:x.lower())


# In[140]:


new_df['tags'][0]


# In[145]:


# Applying stamming :
# loving , lavable , loved is converted to single word love .


# In[146]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[147]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[150]:


stem('in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d samworthington zoesaldana sigourneyweaver jamescameron')


# In[152]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[153]:


# Converting the text to vector :  using countvectorization

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[154]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[155]:


vectors


# In[156]:


cv.get_feature_names()                  # the 5000 words which are present in the coorpes


# In[157]:


# Calculating distance bewteen the vector :  
# It helps to calculate similarity between the movies
# As the distance is less the similarity increases or vice-versa.


# In[158]:


# Using coisine- similarity :

from sklearn.metrics.pairwise import cosine_similarity


# In[159]:


similarity = cosine_similarity(vectors)            # if similarity is 1 its high if it is 0 then it is low 


# In[160]:


list(enumerate(similarity[0]))


# In[161]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[171]:


def recommend(movie):
    movie_index = new_df[new_df['title']== movie].index[0]
    distances = similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(i[0])


# In[172]:


recommend('Batman Begins')


# In[173]:


new_df.iloc[1216].title


# In[176]:


import pickle 

pickle.dump(new_df.to_dict(), open('movies_dict.pkl', 'wb'))


# In[178]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




