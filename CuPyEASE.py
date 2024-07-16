import sys
import torch
import pandas as pd
import cupy as cp

class CuPyEASE:
    def __init__(
        self, train, user_col="user_id", item_col="item_id", score_col=None, reg=250.0
    ):
        """
        :param train: Training DataFrame of user, item, score(optional) values
        :param user_col: Column name for users
        :param item_col: Column name for items
        :param score_col: Column name for scores. Implicit feedback otherwise
        :param reg: Regularization parameter.
                    Change by orders of magnitude to tune (2e1, 2e2, ...,2e4)
        """
        print("Building user + item lookup")
        self.reg = reg
        self.user_col = user_col
        self.item_col = item_col

        self.user_id_col = user_col + "_id"
        self.item_id_col = item_col + "_id"

        self.user_lookup = self.generate_labels(train, self.user_col)
        self.item_lookup = self.generate_labels(train, self.item_col)

        self.item_map = {}
        print("Building item hashmap")
        for _item, _item_id in self.item_lookup.values:
            self.item_map[_item_id] = _item

        train = pd.merge(train, self.user_lookup, on=[self.user_col])
        train = pd.merge(train, self.item_lookup, on=[self.item_col])
        print("User + item lookup complete")
        self.indices = cp.asarray(train[[self.user_id_col, self.item_id_col]].values, dtype=cp.int32).T

        if not score_col:
            self.values = cp.ones(self.indices.shape[1], dtype=cp.float32)
        else:
            self.values = cp.asarray(train[score_col], dtype=cp.float32)

        self.sparse = cp.sparse.coo_matrix((self.values, self.indices))

        print("Sparse data built")

    def generate_labels(self, df, col):
        dist_labels = df[[col]].drop_duplicates()
        dist_labels[col + "_id"] = dist_labels[col].astype("category").cat.codes
        return dist_labels

    def fit(self):
        print("Building G Matrix")
        dense_matrix = self.sparse.toarray()
        G = cp.dot(dense_matrix.T, dense_matrix)
        G += cp.eye(G.shape[0], dtype=cp.float32) * self.reg

        P = cp.linalg.inv(G)

        print("Building B matrix")
        B = P / (-1 * cp.diag(P))
        B = B + cp.eye(B.shape[0], dtype=cp.float32)

        self.B = B

        return

    def predict_all(self, pred_df, k=5, remove_owned=True):
        """
        :param pred_df: DataFrame of users that need predictions
        :param k: Number of items to recommend to each user
        :param remove_owned: Do you want previously interacted items included?
        :return: DataFrame of users + their predictions in sorted order
        """
        pred_df = pred_df[[self.user_col]].drop_duplicates()
        n_orig = pred_df.shape[0]
    
        # Alert to number of dropped users in prediction set
        pred_df = pd.merge(pred_df, self.user_lookup, on=[self.user_col])
        n_curr = pred_df.shape[0]
        if n_orig - n_curr:
            print(
                "Number of unknown users from prediction data = %i" % (n_orig - n_curr)
            )
    
        _output_preds = []
        _score_preds = []
        # Select only user_ids in our user data
        _user_tensor = self.sparse.todense()[cp.asarray(pred_df[self.user_id_col].values, dtype=cp.int32)]
    
        # Make our (raw) predictions
        _preds_tensor = cp.dot(_user_tensor, self.B)
        print("Predictions are made")
        if remove_owned:
            # Discount these items by a large factor (much faster than list comp.)
            print("Removing owned items")
            _preds_tensor += -100.0 * _user_tensor
    
        print("TopK selected per user")
        for _preds in _preds_tensor:
            # Very quick to use .argpartition() vs. argmax()
            top_k_indices = cp.argpartition(_preds, -k)[-k:]
            top_k_scores = _preds[top_k_indices]
            sorted_top_k_indices = top_k_indices[cp.argsort(-top_k_scores)]
            sorted_top_k_scores = _preds[sorted_top_k_indices]
    
            _output_preds.append(
                [self.item_map[int(_id)] for _id in sorted_top_k_indices.tolist()]
            )
            _score_preds.append(
                [float(_v) for _v in sorted_top_k_scores.tolist()]
            )
    
        pred_df["predicted_items"] = _output_preds
        pred_df['score'] = _score_preds
    
        return pred_df

    def score_predictions(self):
        # TODO: Implement this with some common metrics
        return None