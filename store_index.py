import os
import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm
try:
    from pinecone import Pinecone
except ImportError:
    from pinecone.pinecone import Pinecone
from src.helper import download_hugging_face_embedding

load_dotenv()

# 1. Setup
embeddings = download_hugging_face_embedding()
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
index_name = "gamerecommendationsystem"
index = pc.Index(index_name)

# 2. Load Data (Hugging Face parquet)
df = pd.read_parquet(
    "hf://datasets/FronkonGames/steam-games-dataset/data/train-00000-of-00001.parquet",
    storage_options={"token": None},
)

def _first_present(row, keys):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return ""

def _best_description(row, keys):
    best = ""
    for key in keys:
        if key in row and row[key] is not None:
            value = str(row[key]).strip()
            if len(value) > len(best):
                best = value
    return best

MAX_METADATA_BYTES = 38000

def _truncate_bytes(text: str, max_bytes: int = MAX_METADATA_BYTES) -> str:
    if not text:
        return ""
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    return data[:max_bytes].decode("utf-8", errors="ignore")

# Combine Name, Description, and Genres into one searchable string
def create_text(row):
    title = str(_first_present(row, ["Name", "name", "title"]))
    description = _best_description(
        row,
        [
            "About the game",
            "about_the_game",
            "detailed_description",
            "description",
            "short_description",
        ],
    )
    genres = _first_present(row, ["Genres", "genres", "genre"]) 
    if isinstance(genres, (list, tuple)):
        genres = ", ".join(map(str, genres))
    else:
        genres = str(genres)

    combined = f"Game Title: {title.strip()}. Genres: {genres.strip()}. Description: {description.strip()}."
    return _truncate_bytes(combined)

# Apply to full dataset
print("Preparing all games from HF parquet...")
df_test = df.copy()
df_test["combined_text"] = df_test.apply(create_text, axis=1)

# 3. Clear and Upload
# NOTE: Uncomment the next two lines only if you want to refresh the entire database.
# print("Wiping old index data...")
# index.delete(delete_all=True)

print(f"Uploading {len(df_test)} games with Title + Description metadata...")
batch_size = 100
for i in tqdm(range(0, len(df_test), batch_size)):
    batch_df = df_test.iloc[i : i + batch_size]
    vectors_to_upsert = []

    for idx, row in batch_df.iterrows():
        vector = embeddings.embed_query(row["combined_text"])
        vectors_to_upsert.append({
            "id": f"test-{idx}",
            "values": vector,
            "metadata": {"text": row["combined_text"]}
        })
    index.upsert(vectors=vectors_to_upsert)

print("\nReady for full dataset indexing!")