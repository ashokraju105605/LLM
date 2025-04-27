import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSeq2SeqLM
 
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Model to use in retriever
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# Model to use in generator
gen_tokenizer = AutoTokenizer.from_pretrained("t5-small")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
 
def generate_embedding(docs, model, tokenizer):
    # Tokenize each text and convert to PyTorch tensors
    inputs = tokenizer(docs, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
 
    # Embedding defined as mean pooling of all tokens
    attention_mask = inputs["attention_mask"]
    embeddings = outputs.last_hidden_state
 
    expanded_mask = attention_mask.unsqueeze(-1).expand(embeddings.shape).float()
    sum_embeddings = torch.sum(embeddings * expanded_mask, axis=1)
    sum_mask = torch.clamp(expanded_mask.sum(axis=1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
 
    # Convert to numpy array
    return mean_embeddings.cpu().numpy()
 
def retrieve_documents(query, index, documents, k=3):
    # Generate embedding for the query
    query_embedding = generate_embedding(query, model, tokenizer)   # 1xD matrix
    # Search the index for similar documents
    distances, indices = index.search(query_embedding, k)  # 1xk matrices
    # Return the retrieved documents and their distances
    retrieved_docs = [(documents[idx], float(distances[0][i])) for i, idx in enumerate(indices[0])]
    return retrieved_docs
 
def generate_response(query, retrieved_docs, max_length=150):
    # Combine the query and retrieved documents into a single prompt
    if retrieved_docs:
        context = "\n".join(retrieved_docs)
        prompt = f"question: {query} context: {context}"
    else:
        prompt = f"question: {query}"
 
    # Generate a response
    inputs = gen_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = gen_model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    response = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
 
def rag_pipeline(query, documents, retriever_k=3, max_length=150):
    retrieved_docs = retrieve_documents(query, index, documents, k=retriever_k)
    docs = [doc for doc, distance in retrieved_docs]
    response = generate_response(query, docs, max_length=max_length)
    return response, retrieved_docs
 
# Sample document collection
documents = [
    "Transformers are a type of deep learning model introduced in the paper 'Attention "
        "Is All You Need'.",
    "BERT (Bidirectional Encoder Representations from Transformers) is a "
        "transformer-based model designed to understand the context of a word based on "
        "its surroundings.",
    "GPT (Generative Pre-trained Transformer) is a transformer-based model designed for "
        "natural language generation tasks.",
    "T5 (Text-to-Text Transfer Transformer) treats every NLP problem as a text-to-text "
        "problem, where both the input and output are text strings.",
    "RoBERTa is an optimized version of BERT with improved training methodology and more "
        "training data.",
    "DistilBERT is a smaller, faster version of BERT that retains 97% of its language "
        "understanding capabilities.",
    "ALBERT reduces the parameters of BERT by sharing parameters across layers and using "
        "embedding factorization.",
    "XLNet is a generalized autoregressive pretraining method that overcomes the "
        "limitations of BERT by using permutation language modeling.",
    "ELECTRA uses a generator-discriminator architecture for more efficient pretraining.",
    "DeBERTa enhances BERT with disentangled attention and an enhanced mask decoder."
]
 
# Generate embeddings for all documents, then create FAISS index for efficient similarity search
document_embeddings = generate_embedding(documents, model, tokenizer)
dimension = document_embeddings.shape[1]   # Dimension of the embeddings
index = faiss.IndexFlatL2(dimension)       # Using L2 (Euclidean) distance
index.add(document_embeddings)             # Add embeddings to the index
print(f"Created index with {index.ntotal} documents")
 
# Example queries
queries = [
    "What is BERT?",
    "How does GPT work?",
    "What is the difference between BERT and GPT?",
    "What is a smaller version of BERT?"
]
# Run the RAG pipeline for each query
for query in queries:
    response, retrieved_docs = rag_pipeline(query, documents)
    print(f"Query: {query}")
    print()
    print("Retrieved Documents:")
    for i, (doc, distance) in enumerate(retrieved_docs):
        print(f"Document {i+1} (Distance: {distance:.4f}):")
        print(doc)
    print()
    print("Generated Response:")
    print(response)
    print("-" * 20)