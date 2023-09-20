import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, TFAutoModel

def concat_text(text):

  return {
      "text":str(text["RETAILER"])
        + " \n "
        + str(text["BRAND"])
        + " \n "
        + str(text["PRODUCT_CATEGORY"])
        + " \n "
        + str(text["IS_CHILD_CATEGORY_TO"])
  }

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list, model, tokenizer):

    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="tf"
    )
    encoded_input = {k: v for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)

def generate_embeddings(offers_dataset):

    model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = TFAutoModel.from_pretrained(model_ckpt, from_pt=True)
    embeddings_dataset = offers_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"],model,tokenizer).numpy()[0]}
    )
    embeddings_dataset.save_to_disk("data/processed/embeddings.hf")
    return embeddings_dataset

def generate_dataset(offer_data):

    offers_dataset = Dataset.from_pandas(offer_data)
    offers_dataset = offers_dataset.map(concat_text)
    return offers_dataset

def combined_dataset(brand_data, category_data, offer_data):

    df_offer_and_brands = df_offer_retailer.merge(df_brand_category,on='BRAND',how='inner')
    df_offer_brands_categories = df_offer_and_brands.merge(df_categories,left_on='BRAND_BELONGS_TO_CATEGORY',right_on='PRODUCT_CATEGORY',how='inner')
    df_offer_brands_categories = df_offer_brands_categories[['OFFER','RETAILER','BRAND','PRODUCT_CATEGORY','IS_CHILD_CATEGORY_TO']]
    df_offer_brands_categories.to_csv('combined_dataset.csv')
    return df_offer_brands_categories

if __name__ == "__main__":
    
    df_brand_category = pd.read_csv('./data/raw/brand_category.csv')
    df_categories = pd.read_csv('./data/raw/categories.csv')
    df_offer_retailer = pd.read_csv('./data/raw/offer_retailer.csv')

    combined_data = combined_dataset(df_brand_category, df_categories, df_offer_retailer)
    dataset = generate_dataset(combined_data)
    embeddings_dataset = generate_embeddings(dataset)