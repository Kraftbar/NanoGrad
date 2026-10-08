# Local MovieLens-1M Data (NCF split)

MovieLens-1M (GroupLens): 1,000,209 ratings of 3,706 movies by 6,040
users, as split by the Neural Collaborative Filtering authors (He et al.,
2017): every user's latest rating is held out to test, the rest train.
Each test user also has 99 movies they never rated, to rank against.

- `ml-1m.train.rating`: `user  item  rating  timestamp`, 994,169 rows
- `ml-1m.test.rating`: one held-out `user  item  rating  timestamp` per user
- `ml-1m.test.negative`: `(user,item)` then the 99 unrated items

```bash
for f in train.rating test.rating test.negative; do
  curl -L -o data/ml-1m/ml-1m.$f \
    https://raw.githubusercontent.com/hexiangnan/neural_collaborative_filtering/master/Data/ml-1m.$f
done
```
