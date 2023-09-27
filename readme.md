# Data Science Take-Home

## Overview

This repository contains the code for a take-home challenge for the role of Data Scientist, specializing in Natural Language Processing (NLP), at Fetch.

## Problem Statement

The task is to work with a dataset of offers and associated metadata related to the retailers and brands sponsoring these offers. Additionally, there's a dataset listing the brands supported on the platform, along with their corresponding product categories.

## Hosted URL

## Running the code

To create a virtual environment and install the required dependencies, follow these steps:

1. To create a virtual environment and install the required dependencies, follow these steps:

   ```bash
   python -m venv myenv  # Create a virtual environment named 'myenv'

   source myenv/bin/activate  # Activate the virtual environment (Linux/OS X)
   
   pip install -r requirements.txt

   streamlit run your_app.py 


## Approach

In this project, I've adopted a semantic search approach. We'll be generating embeddings for user queries and utilizing FAISS to perform efficient similarity searches within an embeddings database, ultimately returning the k-nearest neighbors.

## Steps

### 1) Dataset Preparation

We have three primary CSV files:

a) `brand_category`
b) `categories`
c) `offer_retailer`

To facilitate semantic search, I merge these dataframes, consolidating information about offers, associated retailers, brands, and respective categories.

### 2) Text Representation

To convert structured data into text-based representations for embedding, each row is transformed into a line of text. We concatenate relevant text from specific columns, generating meaningful text related to each offer.

### 3) Embedding Generation

Next, we convert the textual representations into embeddings using Sentence BERT, allowing for efficient computation of similarities.

### 4) Retrieval

When a user submits a query, we convert it into an embedding using Sentence BERT. Leveraging FAISS indexing for semantic search, we retrieve the k-nearest neighbors based on similarity scores. We then present these neighbors along with their respective similarity scores and associated offers.

### 5) User Interface

For a seamless user experience, we have developed a Proof of Concept (POC) application using Streamlit. Users can enter a product query, and the application will display the top 5 most relevant offers based on the query.

## Screen Shots

![](https://file%2B.vscode-resource.vscode-cdn.net/Users/igautam/Documents/Fetch_Project/fetch/Screen%20Shot%202023-09-19%20at%202.55.18%20PM.png?version%3D1695153489082)
