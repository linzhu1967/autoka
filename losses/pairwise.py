import torch


class InBatchPairwiseNLL:
    """in batch negatives version
    """

    def __init__(self):
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def __call__(self, out_d):
        in_batch_scores, neg_scores = out_d["pos_score"], out_d["neg_score"]
        nb_columns = in_batch_scores.shape[1]
        nb_gpus = int(in_batch_scores.shape[0] / nb_columns)
        temp = torch.cat([in_batch_scores, neg_scores], dim=1)  # concat neg score from BM25 sampling: torch.Size([64, 33])
        # shape (batch_size, batch_size/nb_gpus + 1)
        scores = self.logsoftmax(temp)
        res = -scores[torch.arange(in_batch_scores.shape[0]), torch.arange(nb_columns).repeat(nb_gpus)]
        res = torch.mean(res)
        return res

