import pandas as pd
from datasets import load_from_disk
from transformers import AutoTokenizer, TFAutoModel
from fetch_embedding_prep import get_embeddings
import gradio as gr

def perform_search(query, dataset):
    model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = TFAutoModel.from_pretrained(model_ckpt, from_pt=True)

    question_embedding = get_embeddings([query], model, tokenizer).numpy()
    dataset.add_faiss_index(column="embeddings")
    scores, samples = dataset.get_nearest_examples("embeddings", question_embedding, k=5)

    return scores, samples

def format_results(scores, samples):
    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["scores"] = scores
    samples_df.sort_values("scores", ascending=False, inplace=True)
    samples_df.drop('embeddings', axis=1, inplace=True)
    desired_column_order = ['OFFER', 'scores', 'RETAILER', 'BRAND', 'PRODUCT_CATEGORY', 'IS_CHILD_CATEGORY_TO']
    samples_df = samples_df[desired_column_order]
    return samples_df.to_dict(orient='records')

def search_interface(query):
    # Load the embeddings dataset
    dataset = load_from_disk("/Users/igautam/Documents/Fetch_Project/fetch/data/processed/embeddings.hf")

    if query:
        scores, samples = perform_search(query, dataset)
        result_dict = format_results(scores, samples)
        return result_dict
    else:
        return []

iface = gr.Interface(
    fn=search_interface,
    inputs=gr.inputs.Textbox(),
    outputs=gr.outputs.JSON(),
    live=True,
    capture_session=True
)

if __name__ == "__main__":
    iface.launch()
