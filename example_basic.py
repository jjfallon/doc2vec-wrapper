#------------------------------------------------------
#
# Basic example using the doc2vec_wrapper
#   this uses 
#           - python 2.7
#           - sklearn 0.18.1
#           - gensim 1.0.1
#
#------------------------------------------------------


from sklearn.datasets import fetch_20newsgroups
from doc2vec_wrapper import Doc2VecTransformer


# Download example data
text_data = fetch_20newsgroups(
                categories=['sci.space'], 
                random_state=42, 
                remove=('headers', 'footers', 'quotes')
                ).data

print text_data[0]

# Define model
model = Doc2VecTransformer(min_count=5, size=50, tokeniser="auto", iter=15)

# Fit the model
model.fit(text_data)

# Or fit and transform the model
vectors = model.fit_transform(text_data)
print ""
print vectors[0]

# Transform new data
transformed = model.transform("a new lot of text data")
print ""
print transformed[0]

