import os
from dotenv import load_dotenv

import qdrant_client
from openai import OpenAI

from swarm import Agent
from swarm.repl import run_demo_loop


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize connections
client = OpenAI(api_key=OPENAI_API_KEY)
qdrant = qdrant_client.QdrantClient(host="localhost")

# Set embedding model
EMBEDDING_MODEL = "text-embedding-3-small"

# Set qdrant collection
collection_name = "doc_chat"


def query_qdrant(query, collection_name, vector_name="article", top_k=5):
    # Creates embedding vector from user query
    embedded_query = (
        client.embeddings.create(
            input=query,
            model=EMBEDDING_MODEL,
        )
        .data[0]
        .embedding
    )

    query_results = qdrant.search(
        collection_name=collection_name,
        query_vector=(vector_name, embedded_query),
        limit=top_k,
    )

    return query_results


def query_docs(query):
    """Query the knowledge base for relevant articles."""
    print(f"Searching knowledge base with query: {query}")
    query_results = query_qdrant(query, collection_name="doc_chat")
    output = ""

    for i, article in enumerate(query_results):
        text = article.payload["text"]

        output += text

    if output:
        response = f"Content: {output}"
        return {"response": response}
    else:
        print("No results")
        return {"response": "No results found."}


def generate_completion(prompt):
    response = client.chat_completion.create(
        model="gpt-4", messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def analyze_docs(response):
    prompt = f"Analyze the following content and extract the valuable information from it:\n\n{response}"
    analysis = generate_completion(prompt)

    return {"analysis": analysis}


def create_draft(analysis):
    prompt = f"Given the below refined content, generate a draft from it, with about 3 to 4 paragrpahs: \n\n{analysis}"
    draft = generate_completion(prompt)

    return {"draft": draft}


def structure_draft(draft):
    prompt = f"Given the draft, you need to structure the content into a refined and human readable format and in a pointer format:\n\n{draft}"
    structured = generate_completion(prompt)

    return {"structured": structured}


def handoff_to_analyser():
    return analysis_agent


def handoff_to_draft_generator():
    return draft_agent


def handoff_to_parse_agent():
    return structurize_agent


def parse_answer(structured):
    return {"response": structured}


ui_agent = Agent(
    name="User Interface Agent",
    instructions="Manages user interactions and collecting queries. Call this agent to handle the usual interactions with the user.\
        This agent acts as a User Interface.",
    functions=[query_docs, handoff_to_analyser],
)

analysis_agent = Agent(
    name="Analysis Agent",
    instructions="You are an Agent that analyzes the content, filters out the most important parts of the content, etc..",
    functions=[analyze_docs, handoff_to_draft_generator],
)

draft_agent = Agent(
    name="Draft Generator Agent",
    instructions="You are a draft generator agent which creates a draft from the filtered content.",
    functions=[create_draft, handoff_to_parse_agent],
)

structurize_agent = Agent(
    name="Structured Output Generator Agent",
    instructions="You are a final answer generator agent, whcih takes in a draft and models it in a human readable format",
    functions=[parse_answer],
)


if __name__ == "__main__":
    run_demo_loop(ui_agent)
