# graphity_detection

## how to run
`python3 main.py -i [binary_path] -o [output_path]`

## param parser
- input binary: `-i <path>`, `--input-path <path>`
- output (record): `-o <path>`, `--output-path <path>` 
- model: `-m <model>`, `--model <model>` (optional)
  - rf(default), mlp, knn, svm, lr
- Malware Detection / Family Classification (optional)
  - do nothing if you wanna do malware detection(binary clf)
  - add `-c` if you wanna do family classification
- e.g. `python3 main.py -i testingBin/0021eaf2 -o myDetector_FC_records.csv -m rf -c`
  - use trained rf family classification model to predict '0021eaf2' file and write the result to 'myDetector_FC_records.csv'

## file 
- main.py: detector 、 classifier
- train_model.py: for model training
- data_generate.py: for feature extracting
- benign_graphtheory.csv 、 malware_graphtheory.csv: extracted feature value
- ./Model_classification: classification models
- ./Model_detection: detection models

## Features & Model performance
- number of feature: 23
- detection model accuarcy: 98.06% (RF)
- classification model accuarcy: 97.52% (RF)
