import pandas as pd
import gzip
import json

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('Clothing_Shoes_and_Jewelry_5.json.gz')
selected_columns = df[["reviewText", "overall"]]
selected_columns.to_csv("amazon_review.csv", index=False)


### Data Processing for Fibonacci with letters ###
# df = pd.read_csv("noisy_fibonacci_letters_1M.csv")
# selected_alphabet_columns = df[["context_letters","target_letters"]].copy()
# selected_alphabet_columns.loc[:, "context_letters"] = selected_alphabet_columns["context_letters"].astype(str).apply(lambda row: ",".join(row.split()))
# selected_alphabet_columns.loc[:,"target_letters"] = selected_alphabet_columns["target_letters"].astype(str)
# selected_alphabet_columns.to_csv("processed_noisy_fibonacci_letters_1M.csv", index = False)

### Data Processing for Fibonacci ranging from -100 to 100 as starting two numbers ###
# df = pd.read_csv("noisy_fibonacci_mixed_5to8_1M_seed_-100_100.csv")
# selected_columns = df[["context","target"]].copy()
# selected_columns["target"] = selected_columns["target"].astype(object)
# selected_columns["target"] = selected_columns["target"].astype(str)
# selected_columns.to_csv("processed_noisy_fibonacci_mixed_5to8_1M_seed_-100_100.csv", index = False)