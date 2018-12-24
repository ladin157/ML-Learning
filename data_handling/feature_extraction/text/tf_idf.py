import string
import math

tokenize = lambda doc: doc.lower().split(" ")

document_0 = "China has a strong economy that is growing at a rapid pace. However politically it differs greatly from the US Economy."
document_1 = "At last, China seems serious about confronting an endemic problem: domestic violence and corruption."
document_2 = "Japan's prime minister, Shinzo Abe, is working towards healing the economic turmoil in his own country for his view on the future of his people."
document_3 = "Vladimir Putin is working hard to fix the economy in Russia as the Ruble has tumbled."
document_4 = "What's the future of Abenomics? We asked Shinzo Abe for his views"
document_5 = "Obama has eased sanctions on Cuba while accelerating those against the Russian Economy, even as the Ruble's value falls almost daily."
document_6 = "Vladimir Putin was found to be riding a horse, again, without a shirt on while hunting deer. Vladimir Putin always seems so serious about things - even riding horses."

all_documents = [document_0, document_1, document_2, document_3, document_4, document_5, document_6]

tokenized_documents = [tokenize(d) for d in all_documents]
all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])


# print(all_tokens_set)

# Jaccard similarity
def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection) / len(union)


# Compareing document_2 and document_4
jaccard_index = jaccard_similarity(tokenized_documents[2], tokenized_documents[4])
print(jaccard_index)

intersection = set(tokenized_documents[2]).intersection(set(tokenized_documents[4]))
print(intersection)

jaccard_index = jaccard_similarity(tokenized_documents[1], tokenized_documents[6])
print(jaccard_index)

intersection = set(tokenized_documents[1]).intersection(set(tokenized_documents[6]))
print(intersection)


def term_frequency(term, tokenized_document):
    return tokenized_document.count(term)


def sublinear_term_frequency(term, tokenized_document):
    return 1 + math.log(tokenized_document.count(term))


def augmented_term_frequency(term, tokenized_document):
    max_count = max([term_frequency(t, tokenized_document) for t in tokenized_document])
    return (0.5 + ((0.5 * term_frequency(term, tokenized_document)) / max_count))


def inverse_document_frequencies(tokenized_documents):
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        idf_values[tkn] = 1 + math.log(len(tokenized_documents) / sum(contains_token))
    return idf_values

idf_values = inverse_document_frequencies(tokenized_documents)
print(idf_values['obama'])

print(idf_values['the'])

def tfidf(documents):
    tokenized_documents = [tokenize(d) for d in documents]
    idf = inverse_document_frequencies(tokenized_documents)
    tfidf_documents = []
    for document in tokenized_documents:
        doc_tfidf = []
        for term in idf.keys():
            tf = sublinear_term_frequency(term, document)
            doc_tfidf.append(tf*idf[term])
        tfidf_documents.append(doc_tfidf)
    return tfidf_documents

tfidf_representation = tfidf(all_documents)
print(tfidf_representation[0], document_0)

print('Using scikit-learn')
from sklearn.feature_extraction.text import TfidfVectorizer

sklearn_tfidf = TfidfVectorizer(norm='l2', min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
sklearn_representation = sklearn_tfidf.fit_transform(all_documents)
print(tfidf_representation[0])
print(sklearn_representation.torray()[0].tolist())
print(document_0)