import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer as tfidv
from sklearn.metrics.pairwise import linear_kernel as linker

book = pd.read_csv("C:\\Users\\jzsim\\Downloads\\book1.csv",encoding= 'latin-1')
book.shape
book.columns
book = book.iloc[:,1:]



# Creating a Tfidf Vectorizer to remove all stop words
tfidf = tfidv(stop_words="english")    #taking stop words from tfid vectorizer 
#checking for null
book["Book.Title"].isnull().sum() 

# Preparing the Tfidf matrix by fitting and transforming
#Transform a count matrix to a normalized tf or tf-idf representation

tfidf_matrix = tfidf.fit_transform(book["Book.Title"])
tfidf_matrix.shape #10000, 11435

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linker(tfidf_matrix,tfidf_matrix)

# creating a mapping of anime name to index number 
book_index = pd.Series(book.index,index=book['Book.Title']).drop_duplicates()
book_index["American Pastoral"]

def get_books(Name,topN):
    #topN = 10
    # Getting the book index using its title 
    book_id = book_index[Name]
    
    # Getting the pair wise similarity score for all the book with that book 
    cosine_scores = list(enumerate(cosine_sim_matrix[book_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores,key=lambda x:x[1],reverse = True)
    
    # Get the scores of top 10 most similar book
    cosine_scores_10 = cosine_scores[0:topN+1]
    
    # Getting the books by index 
    book_idx  =  [i[0] for i in cosine_scores_10]
    book_scores =  [i[1] for i in cosine_scores_10]
    
    # Similar book
    book_same = pd.DataFrame(columns=["name","Score"])
    book_same["name"] = book.loc[book_idx,"Book.Title"]
    book_same["Score"] = book_scores
    book_same.reset_index(inplace=True)  
    book_same.drop(["index"],axis=1,inplace=True)
    print (book_same)
    
# Enter your book and number of books to be recommended 
get_books("American Pastoral",topN=10)
get_books("Timeline",topN=5)
get_books("Haveli (Laurel Leaf Books)",topN=15)
get_books("Goodbye, My Little Ones: The True Story of a Murderous Mother and Five Innocent Victims",topN=20)
get_books("My Cat Spit McGee",topN=3)
get_books("Move Your Stuff, Change Your Life : How to Use Feng Shui to Get Love, Money, Respect and Happiness",topN=50)
get_books("The World's Shortest Stories: Murder, Love, Horror, Suspense, All This and Much More in the Most Amazing Short Stories Ever Written, Each One Just 55 Words Long",topN=11)
