import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, TFAutoModel
from datasets import load_from_disk
import streamlit as st
from fetch_embedding_prep import get_embeddings

def perform_search(query, dataset):

    model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = TFAutoModel.from_pretrained(model_ckpt, from_pt=True)

    question_embedding = get_embeddings([query], model, tokenizer).numpy()
    dataset.add_faiss_index(column="embeddings")
    scores, samples = dataset.get_nearest_examples(
    "embeddings", question_embedding, k=5
    )

    return scores,samples

def format_results(scores, samples):

    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["scores"] = scores
    samples_df.sort_values("scores", ascending=False, inplace=True)
    samples_df.drop('embeddings',axis=1,inplace=True)
    desired_column_order = ['OFFER', 'scores', 'RETAILER', 'BRAND', 'PRODUCT_CATEGORY', 'IS_CHILD_CATEGORY_TO']
    samples_df = samples_df[desired_column_order]
    return samples_df


def main():
    st.title("Search Engine")

    # Load the embeddings dataset
    dataset = load_from_disk("/data/processed/embeddings.hf")

    # Text input for user to enter a query
    user_query = st.text_input("Enter your query:")

    if st.button("Search"):
        if user_query:
            scores,samples = perform_search(user_query, dataset)
            result_df = format_results(scores, samples)
            st.write("Search Results:")
            st.write("Row Data:")
            st.write(result_df)
            print(result_df)
        else:
            st.warning("Please enter a query.")

    

if __name__ == "__main__":
    main()



