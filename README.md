# Phishing HTML Detection with SetFit

## Dataset
### training
- phish : 2621
- not phish : 1729
- total : 4350
### validation
- phish : 638
- not phish : 406
- total : 1044
### JSON Dataset path
```angular2html
src/preprocessing/dataset/dataset.json
```
### Make Dataset
```bash
python3 src/preprocessing/dataset/make_dataset.py
pyhton3 src/preprocessing/dataset/parse_dataset_text.py
```
