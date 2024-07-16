import numpy as np

def hit_rate_k(pred_df, list_read_articles_col="item", list_recommended_articles_col="articles_to_recommend"):
    pred_df['hit_rate'] = None
    hit_count = 0 # quantidade de acertos
    read_count = 0 # quantidade de artigos lidos
    for idx, row in pred_df.iterrows(): # itera sobre cada usuário

        iterable = hasattr(row[list_read_articles_col], '__iter__')

        aux = hit_count
        for read_article in row[list_read_articles_col] if iterable else [row[list_read_articles_col]]: # itera sobre cada artigo lido pelo usuário
            # verifica se o artigo lido está dentro da lista de artigos recomendados
            if read_article in row[list_recommended_articles_col]: 
                hit_count += 1 
            read_count += 1
        pred_df.loc[idx, 'hit_rate'] = (hit_count - aux)/len(row[list_read_articles_col])

    hit_rate = hit_count/read_count
    return hit_rate

def calculate_ndcg(pred_df, list_read_articles_col="item", list_recommended_articles_col="articles_to_recommend"):
    # referência para implementação: https://github.com/lucky7323/nDCG/blob/master/ndcg.py
    pred_df['ndcg'] = None
    ndgs = []
    for idx, row in pred_df.iterrows():  # itera sobre cada usuário
        # o rank_length garante que a quantidade de artigos lidos é igual a quantidade de artigos recomendados
        rank_length = min(len(row[list_read_articles_col]), len(row[list_recommended_articles_col]))
        discount = 1/np.log2(np.arange(rank_length) + 2)
        dcg = 0
        idcg = 0
        for idx_read, recommended_article in enumerate(row[list_recommended_articles_col][:rank_length]): # itera sobre cada artigo recomendado
            # verifica se o artigo recomendado é relevante (0 ou 1)
            relevance = recommended_article in row[list_read_articles_col]
            dcg += relevance*discount[idx_read] # dcg calculado
            idcg += 1*discount[idx_read] # dcg ideal
                
        # Calcula o DCG normalizado
        if idcg == 0:
            ndcg = 0
        else:
            ndcg = dcg / idcg
        pred_df.loc[idx, 'ndcg'] = ndcg

        ndgs.append(ndcg)
        
    return np.mean(ndgs)