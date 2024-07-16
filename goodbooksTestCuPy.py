import pandas as pd
from CuPyEASE import CuPyEASE
import time
from metrics import hit_rate_k, calculate_ndcg

################################################################################

# Criando o dataset para o treinamento (coluna user, item e rating)

books = pd.read_csv("./books.csv")
books = books[(books["language_code"] == "eng") | (books["language_code"] == "en-US")]
print("\n\nBooks", books)

df = pd.read_csv("./ratings.csv")
df = pd.merge(df, books[["book_id", "original_title"]].dropna(), on=["book_id"], how='inner')
print("\n\nDF", df)

df.to_csv("completeData.csv", index=False)

################################################################################

# Treinamento implicito
te = CuPyEASE(df, user_col="user_id", item_col="original_title")

initial = time.time()
te.fit()
final = time.time()
print(f'Train time out: {final - initial:.6f}')

initial = time.time()
out = te.predict_all(df[['user_id']], k=10)
final = time.time()
print(f'Predict time out: {final - initial:.6f}')

print("\n\nout:", out)

################################################################################

# Treinamento explicito
ter = CuPyEASE(df, user_col="user_id", item_col="book_id", score_col="rating")

initial = time.time()
ter.fit()
final = time.time()
print(f'Train time outr: {final - initial:.2f}')

initial = time.time()
outr = ter.predict_all(df[['user_id']], k=10)
final = time.time()
print(f'Predict time outr: {final - initial:.2f}')

# Metricas (não funciona ainda)
# df_rec_list = outr.merge(
#     df[['user_id', 'book_id']], 
#     on='user_id', 
#     how='left'
# ).rename({'predicted_items':'articles_to_recommend'}, axis=1)
# 
# df_rec_list = df_rec_list.groupby('user_id').agg(
#     item = ('book_id', lambda x: list(x)), 
#     articles_to_recommend=('articles_to_recommend', 'last')
# )
# 
# hr = hit_rate_k(df_rec_list, "item", "articles_to_recommend")
# ndcg = calculate_ndcg(df_rec_list, "item", "articles_to_recommend")
# 
# print("\n\n\ndf_rec_list:", df_rec_list)
# print("\n\n\nHit rate:", hr)
# print("\n\n\nnDCG:", ndcg)

print("\n\noutr:")
print(outr)

# print("\n\ndf:")
# print(df)
