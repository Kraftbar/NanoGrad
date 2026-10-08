# Local AG News Data

AG News (Zhang, Zhao and LeCun, 2015): 120,000 training and 7,600 test
news articles (title and description), in 4 balanced classes: World,
Sports, Business, Sci/Tech. CSV rows are `class (1-4), title, description`.

```bash
for f in train test; do
  curl -L -o data/ag_news/$f.csv \
    https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/$f.csv
done
```
