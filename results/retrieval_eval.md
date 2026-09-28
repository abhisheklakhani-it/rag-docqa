| Retriever | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | ms / question |
|:--|--:|--:|--:|--:|--:|--:|
| tfidf | 0.406 | 0.658 | 0.724 | 0.522 | 0.567 | 6.15 |
| bm25 | 0.531 | 0.735 | 0.806 | 0.633 | 0.670 | 0.21 |
| dense | 0.574 | 0.777 | 0.848 | 0.683 | 0.718 | 7.63 |
| hybrid | 0.580 | 0.794 | 0.855 | 0.693 | 0.728 | 9.70 |

| A | B | nDCG@10 A − B | p-value | Significant (p < 0.05)? |
|:--|:--|--:|--:|:--|
| bm25 | tfidf | +0.103 | < 0.001 | yes |
| dense | bm25 | +0.048 | 0.003 | yes |
| hybrid | tfidf | +0.161 | < 0.001 | yes |
| hybrid | bm25 | +0.058 | < 0.001 | yes |
| hybrid | dense | +0.009 | 0.431 | no |
