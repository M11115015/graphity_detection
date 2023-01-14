# graphity_detection

## how to run & parameters
- python3 main.py -i [binary_path] -o [output_path]
- input binary: `-i <path>`, `--input-path <path>`
- output (record): `-o <path>`, `--output-path <path>` 
- model: `-m <model>`, `--model <model>` (optional)
  - rf(default), mlp, knn, svm, lr
- Malware Detection / Family Classification (optional)
  - do nothing if you wanna do malware detection(binary clf)
  - add -c if you wanna do family classification
- e.g. `python3 main.py -i testingBin/0021eaf2 -o myDetector_FC_records.csv -m rf -c`
  - using trained rf family classifier(`-c`), predict '0021eaf2' and write the result to 'myDetector_FC_records.csv'
  - add -W ignore if you keep getting bothered by warning msg

## file 
- train_model.py: for model training
- data_generate.py: for feature extracting
- benign_graphtheory.csv 、 malware_graphtheory.csv: extracted feature value
- detection model accuarcy: 98.06% (RF)
- classification model accuarcy: 97.52% (RF)
- number of feature: 23
