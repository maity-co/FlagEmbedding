from FlagEmbedding import FlagAutoModel

corpus = [
    "Michael Jackson was a legendary pop icon known for his record-breaking music and dance innovations.",
    "Fei-Fei Li is a professor in Stanford University, revolutionized computer vision with the ImageNet project.",
    "Brad Pitt is a versatile actor and producer known for his roles in films like 'Fight Club' and 'Once Upon a Time in Hollywood.'",
    "Geoffrey Hinton, as a foundational figure in AI, received Turing Award for his contribution in deep learning.",
    "Eminem is a renowned rapper and one of the best-selling music artists of all time.",
    "Taylor Swift is a Grammy-winning singer-songwriter known for her narrative-driven music.",
    "Sam Altman leads OpenAI as its CEO, with astonishing works of GPT series and pursuing safe and beneficial AI.",
    "Morgan Freeman is an acclaimed actor famous for his distinctive voice and diverse roles.",
    "Andrew Ng spread AI knowledge globally via public courses on Coursera and Stanford University.",
    "Robert Downey Jr. is an iconic actor best known for playing Iron Man in the Marvel Cinematic Universe.",
]

query = "Who could be an expert of neural network?"
print(query)

from FlagEmbedding import FlagModel

# get the BGE embedding model
model = FlagModel('BAAI/bge-base-en-v1.5',
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                  use_fp16=True)

# get the embedding of the query and corpus
print("Encoding")
corpus_embeddings = model.encode(corpus)
query_embedding = model.encode(query)

#
# print("shape of the query embedding:  ", query_embedding.shape)
# print("shape of the corpus embeddings:", corpus_embeddings.shape)

# print(query_embedding[:10])

print("scoring")
sim_scores = query_embedding @ corpus_embeddings.T
# print(sim_scores)


# get the indices in sorted order
sorted_indices = sorted(range(len(sim_scores)), key=lambda k: sim_scores[k], reverse=True)
#print(sorted_indices)

print(corpus[3])

# iteratively print the score and corresponding sentences in descending order

# for i in sorted_indices:
#     print(f"Score of {sim_scores[i]:.3f}: \"{corpus[i]}\"")