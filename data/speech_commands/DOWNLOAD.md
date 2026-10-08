# Local Speech Commands Data

Speech Commands v0.02 (Warden, 2018): 105,829 one-second 16 kHz clips of
35 words, with `validation_list.txt` and `testing_list.txt`, plus the
official 12-class test set (4,890 clips) in `test_set/`. (The dataset's
own README.md and LICENSE come with the archive.)

```bash
cd data/speech_commands
curl -L -o sc.tar.gz https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
tar -xzf sc.tar.gz && rm sc.tar.gz
mkdir -p test_set
curl -L -o test.tar.gz https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz
tar -xzf test.tar.gz -C test_set && rm test.tar.gz
```

`kws.py` computes the MFCCs on first use and caches them in `mfcc.pkl`.
