from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from  pinecone import Pinecone
import time
from dotenv import load_dotenv
import os
from langchain_ollama import OllamaLLM

load_dotenv() 
PINECONE_API_KEY =os.environ.get('PINECONE_API_KEY')
index = "gamerecommendationsystem"
llm = OllamaLLM(model="llama2")
## Extract data from CSV file
def load_file(path):
    loader = CSVLoader(file_path=path, encoding="utf-8")
    data = loader.load()
    return data

# Transform data(create chunks)
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20) 
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks

## download model from Hugging face
def download_hugging_face_embedding():
    model_name = "BAAI/bge-large-en"
    encode_kwargs = {'normalize_embeddings': True} 

    embedding = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs=encode_kwargs
    )
    return embedding

embeddings = download_hugging_face_embedding()

pc=Pinecone(PINECONE_API_KEY)
index = pc.Index(index)
limit=3070
def retrieve(query,conversation_history):
    vector=embeddings.embed_query(query)
    # get relevant contexts
    contexts = []
    for message in conversation_history:
        contexts.append(f"{message['role'].capitalize()}: {message['content']}\n")
    l=len(contexts)
    res=index.query(vector=vector,top_k=1,include_values=True,include_metadata=True)
    time.sleep(2)
    for x in res['matches']:
        contexts.append(x['metadata']['text'])

    print(f"Retrieved {len(contexts)} contexts, sleeping for 5 seconds...")
    if len(contexts)<=l:
        print("Timed out waiting for contexts to be retrieved.")
        contexts = ["No contexts retrieved. Try to answer the question yourself!"]

    # Updated prompt_start (The "Guardrails")
    prompt_start = (
        "You are a Game Recommendation Expert. Your goal is to provide accurate information based ONLY on the provided database context.\n\n"
        "RULES:\n"
        "1. If the user is just being social (greeting, small talk, thanking you), Respond to this social message politely as a game expert:\n" \
        "2. Or if they are asking for a game recommendation or game information, then use ONLY the provided Context to answer the question.\n"
        "3. If the context contains a game that matches the user's request (e.g., matching the genre or theme), recommend it. If no match is found, say you don't have information on that yet.\n"        
        "4. When recommending a game, always start by stating the 'Game Title' found in the context.\n"
        "5. Do not invent details. If the context is missing info, don't guess.\n\n"
        "Context:\n"
    )

    # Updated prompt_end (The "Trigger")
    prompt_end = (
        f"\n\nQuestion: {query}\n"
        "Only return the helpful answer based on the rules above and nothing else.\n"
        "Answer:"
    )

    # append contexts until hitting limit
    prompt=""
    print(len(contexts))
    for i in range(0, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts[:i-1]) +
                prompt_end
            )
            break
        elif i == len(contexts)-1:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts) +
                prompt_end
            )
    return prompt

def complete(prompt):
    # Use .invoke() instead of calling the object directly
    response = llm.invoke(prompt)
    return response

def chatbot(query,chat_history):
    query_with_contexts = retrieve(query,chat_history)
    print(query_with_contexts)
    response=complete(query_with_contexts)
    chat_history.append({"role": "user", "content":query})
    chat_history.append({"role": "assistant", "content": response})
    return response,chat_history