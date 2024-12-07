import streamlit as st
import dotenv
import json
import yfinance as yf
import concurrent.futures
import os
import requests
import numpy as np
import time

from pinecone import Pinecone
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore


def get_stock_info(symbol: str) -> dict:
    """
    Retrieves and formats detailed information about a stock from Yahoo Finance.
    Implements caching to reduce repeated requests.
    """

    # Check if we have cached data
    if symbol in stock_cache:
        return stock_cache[symbol]

    # Not cached, proceed with a fresh request
    data = yf.Ticker(symbol)
    try:
        stock_info = data.info
    except Exception as e:
        print(f"Error retrieving info for {symbol}: {e}")
        return None

    if not stock_info or 'symbol' not in stock_info:
        # Potential rate limit or no data found
        print(f"No info available for {symbol}. Possibly rate-limited.")
        return None

    properties = {
        "Ticker": stock_info.get('symbol', 'Information not available'),
        'Name': stock_info.get('longName', 'Information not available'),
        'Business Summary': stock_info.get('longBusinessSummary', 'Information not available'),
        'City': stock_info.get('city', 'Information not available'),
        'State': stock_info.get('state', 'Information not available'),
        'Country': stock_info.get('country', 'Information not available'),
        'Industry': stock_info.get('industry', 'Information not available'),
        'Sector': stock_info.get('sector', 'Information not available')
    }

    # Cache the result
    stock_cache[symbol] = properties
    return properties


def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    """
    Generates embeddings for the given text using a specified Hugging Face model.

    Args:
        text (str): The input text to generate embeddings for.
        model_name (str): The name of the Hugging Face model to use.
                          Defaults to "sentence-transformers/all-mpnet-base-v2".

    Returns:
        np.ndarray: The generated embeddings as a NumPy array.
    """
    model = SentenceTransformer(model_name)
    return model.encode(text)


def cosine_similarity_between_sentences(sentence1, sentence2):
    """
    Calculates the cosine similarity between two sentences.

    Args:
        sentence1 (str): The first sentence for similarity comparison.
        sentence2 (str): The second sentence for similarity comparison.

    Returns:
        float: The cosine similarity score between the two sentences,
               ranging from -1 (completely opposite) to 1 (identical).

    Notes:
        Prints the similarity score to the console in a formatted string.
    """
    # Get embeddings for both sentences
    embedding1 = np.array(get_huggingface_embeddings(sentence1))
    embedding2 = np.array(get_huggingface_embeddings(sentence2))

    # Reshape embeddings for cosine_similarity function
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)

    # Calculate cosine similarity
    similarity = cosine_similarity(embedding1, embedding2)
    similarity_score = similarity[0][0]
    print(f"Cosine similarity between the two sentences: {
          similarity_score:.4f}")
    return similarity_score
    """
    Calculates the cosine similarity between two sentences.

    Args:
        sentence1 (str): The first sentence for similarity comparison.
        sentence2 (str): The second sentence for similarity comparison.

    Returns:
        float: The cosine similarity score between the two sentences,
               ranging from -1 (completely opposite) to 1 (identical).

    Notes:
        Prints the similarity score to the console in a formatted string.
    """
    # Get embeddings for both sentences
    embedding1 = np.array(get_huggingface_embeddings(sentence1))
    embedding2 = np.array(get_huggingface_embeddings(sentence2))

    # Reshape embeddings for cosine_similarity function
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)

    # Calculate cosine similarity
    similarity = cosine_similarity(embedding1, embedding2)
    similarity_score = similarity[0][0]
    print(f"Cosine similarity between the two sentences: {
          similarity_score:.4f}")
    return similarity_score


def get_company_tickers():
    """
    Downloads and parses the Stock ticker symbols from the GitHub-hosted SEC company tickers JSON file.

    Returns:
        dict: A dictionary containing company tickers and related information.

    Notes:
        The data is sourced from the official SEC website via a GitHub repository:
        https://raw.githubusercontent.com/zchisholm/zentiment-analytics/refs/heads/main/company_tickers.json
    """
    # URL to fetch the raw JSON file from GitHub
    url = "https://raw.githubusercontent.com/zchisholm/zentiment-analytics/refs/heads/main/company_tickers.json"

    # Making a GET request to the URL
    response = requests.get(url)

    # Checking if the request was successful
    if response.status_code == 200:
        # Parse the JSON content directly
        company_tickers = json.loads(response.content.decode('utf-8'))

        # Optionally save the content to a local file for future use
        with open("company_tickers.json", "w", encoding="utf-8") as file:
            json.dump(company_tickers, file, indent=4)

        print("File downloaded successfully and saved as 'company_tickers.json'")
        return company_tickers
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
        return None


def process_stock(stock_ticker: str) -> str:
    # Skip if already processed
    if stock_ticker in successful_tickers:
        return f"Already processed {stock_ticker}"

    try:
        # Get and store stock data
        stock_data = get_stock_info(stock_ticker)
        stock_description = stock_data['Business Summary']
        if not stock_data:
            raise ValueError("No stock data retrieved. Possibly rate-limited.")

        # Store stock description in Pinecone
        PineconeVectorStore.from_documents(
            documents=[
                Document(page_content=stock_description, metadata=stock_data)],
            embedding=hf_embeddings,
            index_name=pinecone_index_name,
            namespace=namespace
        )

        # Track success
        with open('successful_tickers.txt', 'a') as f:
            f.write(f"{stock_ticker}\n")
        successful_tickers.append(stock_ticker)

        # Add a delay after successful processing to avoid rate limits
        time.sleep(1)

        return f"Processed {stock_ticker} successfully"

    except Exception as e:
        # Track failure
        with open('unsuccessful_tickers.txt', 'a') as f:
            f.write(f"{stock_ticker}\n")
        unsuccessful_tickers.append(stock_ticker)

        return f"ERROR processing {stock_ticker}: {e}"


def parallel_process_stocks(tickers: list, max_workers: int = 10) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(process_stock, ticker): ticker
            for ticker in tickers
        }

        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                print(result)

                # Stop on error
                if result.startswith("ERROR"):
                    print(f"Stopping program due to error in {ticker}")
                    executor.shutdown(wait=False)
                    raise SystemExit(1)

            except Exception as exc:
                print(f'{ticker} generated an exception: {exc}')
                print("Stopping program due to exception")
                executor.shutdown(wait=False)
                raise SystemExit(1)


def initialize_pinecone():
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    return Pinecone(api_key=pinecone_api_key).Index(pinecone_index_name)


def generate_query_embedding(query, get_huggingface_embeddings):
    """
    Generate embeddings for a given query.

    Parameters:
        query (str): The user's query.
        get_huggingface_embeddings (function): Function to generate embeddings for the query.

    Returns:
        list: The query embeddings as a list.
    """
    return get_huggingface_embeddings(query).tolist()


def query_pinecone_index(embedding, pinecone_index, namespace, top_k=10):
    """
    Query the Pinecone index to retrieve top matches based on embeddings.

    Parameters:
        embedding (list): The query embeddings.
        pinecone_index (object): The Pinecone index object.
        namespace (str): Namespace for Pinecone index search.
        top_k (int): Number of top matches to retrieve.

    Returns:
        list: The top matches including metadata.
    """
    result = pinecone_index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )
    return result['matches']


def construct_augmented_query(query, contexts):
    """
    Construct an augmented query by combining retrieved contexts with the user query.

    Parameters:
        query (str): The user's query.
        contexts (list): The retrieved contexts.

    Returns:
        str: The augmented query string.
    """
    context_text = "\n\n-------\n\n".join(contexts[:10])
    return (
        "<CONTEXT>\n" +
        context_text +
        "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" +
        query
    )


def generate_response(augmented_query, client, model="llama-3.1-70b-versatile"):
    """
    Generate a response using a language model.

    Parameters:
        augmented_query (str): The query with contextual information.
        client (object): The LLM client for generating responses.
        model (str): The LLM model to use.

    Returns:
        str: The response content from the language model.
    """
    system_prompt = "You are an expert at providing answers about stocks. Please answer my question provided."
    llm_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )
    return llm_response.choices[0].message.content


def perform_rag(query, pinecone_index, namespace, get_huggingface_embeddings, client):
    """
    Perform Retrieval-Augmented Generation (RAG) given a query.

    Parameters:
        query (str): The user's query.
        pinecone_index (object): The Pinecone index object.
        namespace (str): Namespace for Pinecone index search.
        get_huggingface_embeddings (function): Function to generate embeddings for the query.
        client (object): The LLM client for generating responses.

    Returns:
        str: The response from the language model (LLM).
    """
    embedding = generate_query_embedding(query, get_huggingface_embeddings)
    matches = query_pinecone_index(embedding, pinecone_index, namespace)
    contexts = [item['metadata']['text'] for item in matches]
    augmented_query = construct_augmented_query(query, contexts)
    return generate_response(augmented_query, client)


# Streamlit Setup
st.title("Welcome to Zentiment Analysis")
st.subheader("Stock Oracle")


# Pinecone Setup
pinecone_index_name = "zentiment-analysis"
namespace = "stock-descriptions"
pinecone_index = initialize_pinecone()
hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = PineconeVectorStore(
    index_name=pinecone_index_name, embedding=hf_embeddings)

# Initialize tracking lists
stock_cache = {}
if os.path.exists('successful_tickers.txt'):
    with open('successful_tickers.txt', 'r') as f:
        successful_tickers = [line.strip() for line in f.readlines()]
else:
    successful_tickers = []
unsuccessful_tickers = []

company_tickers = get_company_tickers()

###
# Prepare your tickers
if company_tickers:
    all_tickers = [company_tickers[num]['ticker']
                   for num in company_tickers.keys()]
    # Filter out tickers already processed
    tickers_to_process = [
        t for t in all_tickers if t not in successful_tickers]
else:
    tickers_to_process = []

# Process them
parallel_process_stocks(tickers_to_process, max_workers=5)

# Initialize OpenAI client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",  # Replace with your actual API URL
    api_key=st.secrets["GROQ_API_KEY"]
)

# Chat Interface
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask about stocks or tickers..."):
    st.session_state["messages"].append({"role": "user", "content": query})
    try:
        response = perform_rag(
            query=query,
            pinecone_index=pinecone_index,
            namespace=namespace,
            get_huggingface_embeddings=get_huggingface_embeddings,
            client=client
        )
        st.session_state["messages"].append(
            {"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"An error occurred: {e}")
