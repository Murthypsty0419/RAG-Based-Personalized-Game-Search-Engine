from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
try:
    from pinecone import Pinecone
except ImportError:
    from pinecone.pinecone import Pinecone
import time
from dotenv import load_dotenv
import os
from langchain_ollama import OllamaLLM
import re

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

def clean_query(query: str) -> str:
    cleaned = query.strip()
    cleaned = re.sub(r"\b\d{1,2}:\d{2}\b", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def detect_intent(query: str) -> str:
    intent_prompt = (
        "Classify this user message into exactly one label: social-chat, game-request, or other.\n"
        "Definitions:\n"
        "- social-chat: greetings, thanks, small talk, capability questions.\n"
        "- game-request: asks for game recommendations or game information.\n"
        "- other: anything else.\n"
        f"Message: {query}\n"
        "Return only one label."
    )
    try:
        label = llm.invoke(intent_prompt).strip().lower()
    except Exception:
        return "other"

    if "social-chat" in label:
        return "social-chat"
    if "game-request" in label:
        return "game-request"
    return "other"

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

    # Game-answer guardrails for retrieval mode.
    prompt_start = (
        "You are a Game Recommendation Expert. Your goal is to provide accurate information based ONLY on the provided database context.\n\n"
        "RULES:\n"
        "1. Use ONLY the provided Context to answer the game-related question.\n"
        "2. If context does not truly support the request, say you do not have enough information yet.\n"
        "3. When recommending a game, start with the exact 'Game Title' from context.\n"
        "4. Do not invent details. If info is missing, don't guess.\n\n"
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
    # If Ollama is not running, return a clear message instead of crashing the request.
    try:
        response = llm.invoke(prompt)
        return response
    except Exception as exc:
        err = str(exc).lower()
        if "connection refused" in err or "failed to connect" in err:
            return (
                "I can retrieve game context, but the local LLM service is not reachable. "
                "Please start Ollama (`ollama serve`) and make sure the model is available "
                "with `ollama pull llama2`."
            )
        raise

def chatbot(query,chat_history):
    query = clean_query(query)
    intent = detect_intent(query)

    if intent == "social-chat":
        social_prompt = (
            "You are a Game Recommendation Expert. "
            "Respond naturally and briefly to this social/capability message. "
            "Do not recommend games unless the user asks for recommendations.\n"
            f"Message: {query}\n"
            "Answer:"
        )
        response = complete(social_prompt)
    elif intent == "game-request":
        query_with_contexts = retrieve(query,chat_history)
        print(query_with_contexts)
        response = complete(query_with_contexts)
    else:
        response = (
            "I can help with game recommendations and game information. "
            "Tell me what type of game you want, for example genre, mood, or theme."
        )

    chat_history.append({"role": "user", "content":query})
    chat_history.append({"role": "assistant", "content": response})
    return response,chat_history