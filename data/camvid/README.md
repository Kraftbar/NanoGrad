# Local CamVid Data

CamVid (Brostow, Fauqueur and Cipolla, 2008) as prepared for SegNet:
701 frames of driving video at 360 x 480, RGB PNGs, with label PNGs whose
pixel values are classes 0-10 (sky, building, pole, road, pavement, tree,
sign, fence, car, pedestrian, bicyclist) or 11 (void). The split files
list 367 training, 101 validation and 233 test pairs.

```bash
B=https://raw.githubusercontent.com/alexgkendall/SegNet-Tutorial/master/CamVid
cd data/camvid
for s in train val test; do curl -sS -o $s.txt $B/$s.txt; mkdir -p $s ${s}annot; done
cat train.txt val.txt test.txt | tr ' ' '\n' | sed 's#^/SegNet/CamVid/##' | grep png > files.txt
xargs -P 8 -I{} curl -sS --retry 3 -o {} "$B/{}" < files.txt
```

`segment.py` decodes the PNGs on first use (several minutes) and caches the
half-size images in `camvid_176x240.pkl`.
