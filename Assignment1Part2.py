# Databricks notebook source
# MAGIC %pip install nltk

# COMMAND ----------

#source link - http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz
plotSummaries = sc.textFile('/FileStore/tables/plot_summaries.txt')
#creating a dataframe, converting it to rdd, and mapping over RDD
moviesMetadata = spark.read.csv("/FileStore/tables/movie_metadata.tsv", sep=r'\t', header=False)
movies = moviesMetadata.rdd 
movieNames = movies.map(lambda row : (row._c0, row._c2))
movies = movieNames.collectAsMap()

# COMMAND ----------

#getting the key-value tupple pairs
summaries = plotSummaries.map(lambda line : line.split('\t')).map(lambda x: (x[0],x[1])).map(lambda x:(x[0],x[1].lower())).map(lambda x : (x[0], x[1].split(" ")))
summaries.take(5)

# COMMAND ----------

import nltk
nltk.download('stopwords')

# COMMAND ----------

#return words which are not stopwords
import string
from nltk.corpus import stopwords
def removeStopWords(words):
    stopWords = list(stopwords.words('english'))
    puncts = string.punctuation+' '
    ans = []
    for word in words:
        word = word.strip(puncts)
        if len(word) == 0:
            continue
        if word not in stopWords:
            ans.append(word)
    return ans

# COMMAND ----------

#get the final list of valid words
def getFinalWords(summary):
    #words = summary.lower().split(" ")
    finalWords = removeStopWords(summary)
    return finalWords

# COMMAND ----------

#update summaries with the list of proper words
summaries=summaries.map(lambda x : (x[0], getFinalWords(x[1])))
summaries.take(5)

#summaries.map(lambda x : (x[0], getFinalWords(x[1])))

# COMMAND ----------

#total number of movies
N = summaries.count()
N

# COMMAND ----------

# calculating tij which is count of ith word in a jth movie document
summary = summaries.flatMapValues(lambda x : x)
tij = summary.map(lambda t : ((t[0], t[1]), 1)).reduceByKey(lambda x, y : x+y)
tij.take(5)

# COMMAND ----------

# calculating count of a ith word in all the movie documents
ni = summary.distinct().map(lambda t : (t[1], t[0])).reduceByKey(lambda x, y : x+" "+y).map(lambda t : (t[0], len(t[1].split(" "))))
ni = ni.collectAsMap()

# COMMAND ----------

import math
tfIdf = tij
tfIdf = tfIdf.map(lambda x : (x[0], float(x[1]*float(math.log(N/ni[x[0][1]]))))).sortBy(lambda x : -x[1])
tfIdf = tfIdf.collectAsMap()

# COMMAND ----------

tfIdf

# COMMAND ----------

movies

# COMMAND ----------

def singleSearch(query, topHitsCount=10):
    output = []
    for values in tfIdf.keys():
        if values[1] == query:
            output.append(values[0])
    ans = []
    for val in range(min(len(output), topHitsCount)):
        ans.append(movies[output[val]])
    print("Results : "+str(ans))

# COMMAND ----------

from scipy import spatial
def multipleSearch(query, topHitsCount=10):
    qcount = {}
    for q in query:
        if q not in qcount.keys():
            qcount[q] = 1
        else:
            qcount[q]+=1
    words = []
    qVector = []
    for key, value in qcount.items():
        words.append(key)
        qVector.append(value)

    movieId = list(movies.keys())
    cosine = {}
    for doc in movieId:
        dVector = []
        for word in words:
            t = (doc, word)
            if t in tfIdf:
                dVector.append(tfIdf[t])
            else:
                dVector.append(0)
        s = set(dVector)
        if len(s) == 1 and 0 in s:
            continue
        result = 1 - spatial.distance.cosine(qVector, dVector)
        cosine[doc] = result
    
    sortedHits = list(sorted(cosine.items(), key=lambda item: -item[1])) # list maintains the order of sort
    sortedHits = sortedHits[0:min(len(sortedHits), topHitsCount)]
    ans=[]
    for k in range(len(sortedHits)):# getting movie names for the resultant movie ids
        temp_key = sortedHits[k][0] # movie id
        #sortedHits[k]= (movies[temp_key], sortedHits[k][1])
        ans.append(movies[temp_key])
    print("Results : "+str(ans))

# COMMAND ----------

query_file = spark.read.csv("/FileStore/tables/queries_1-1.csv") # all search queries stored at first row of the csv file with delimiter','
queries=[]
for i in query_file.collect()[0]:
    queries.append([getFinalWords(i.strip().lower()),i])
for query in queries:
    print("\n")
    print('For Query "'+ query[1]+'"')
    if len(query[0])==0:
        print("Invalid Query")
        continue
    
    if len(query[0])>1:
        multipleSearch(query[0])
    else:
        singleSearch(query[0][0])

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


