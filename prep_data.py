import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import qdrant_client
from openai import OpenAI
from qdrant_client.http import models as rest

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
GPT_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"

docs_list = os.listdir("data")

articles = []


def read_data_from_pdf(path):
    text = ""
    # with open(path, "rb") as f:
    pdf_reader = PdfReader(path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks


for x in docs_list:
    if x != ".DS_Store":
        article = read_data_from_pdf(f"data/{x}")
        articles.append(article)

documents = []

for article in articles:
    for chunk in get_text_chunks(article):
        documents.append({"text": chunk})

for i, x in enumerate(documents):
    try:
        embedding = client.embeddings.create(model=EMBEDDING_MODEL, input=x["text"])
        documents[i].update({"embedding": embedding.data[0].embedding})
    except Exception as e:
        pass

qdrant = qdrant_client.QdrantClient(host="localhost")
print(qdrant.get_collections())

collection_name = "doc_chat"

vector_size = len(documents[0]["embedding"])
print(f"Vector Size : {vector_size}")

article_df = pd.DataFrame(documents)

# article_df.to_csv("./")

# Delete the collection if it exists, so we can rewrite it changes to articles were made
# if qdrant.get_collection(collection_name=collection_name):
try:
    qdrant.delete_collection(collection_name=collection_name)
except Exception as e:
    print("Creating New Collection")

# Create Vector DB collection
qdrant.create_collection(
    collection_name=collection_name,
    vectors_config={
        "article": rest.VectorParams(
            distance=rest.Distance.COSINE,
            size=vector_size,
        )
    },
)

# Populate collection with vectors
try:
    qdrant.upsert(
        collection_name=collection_name,
        points=[
            rest.PointStruct(
                id=k,
                vector={
                    "article": v["embedding"],
                },
                payload=v.to_dict(),
            )
            for k, v in article_df.iterrows()
        ],
    )
except Exception as e:
    pass
