# Regularized Drop with RoBERTa SEP token Pooling

## Dacon competitions
- [문장 유형 분류 AI 경진대회](https://dacon.io/competitions/official/236037/overview/description)

## How to Use

- Run train.py

## Requirements

- `transformers == 4.25.1`
- `pandas == 1.3.5`
- `numpy == 1.21.6`
- `torch == 1.13.1+cu116`
- `scikit-learn == 1.0.2`
- `tqdm == 4.64.1`
- `pyperclip == 1.8.2`
- `selenium == 4.7.2`

## Metric

- weighted F1 score

## Score

- Public score : 75.771 (6/565) (ensemble seed = [3,4,5])
- Private score : 75.27 (18/565)

## Future works

- loss can't coverge with klue/RoBERTa-large (I think it's because of hyperparameter. it can be get higher score)

## Workers

### [노영준](https://github.com/youngjun-99)
- Seed ensemble
- CV ensemble
- Exploratory Data Analysis
- Code refactoring
- Project Managing
- Regularized Dropout
- Undersampling

### [이가은](https://github.com/gaeun5744)
- AEDA
- Jensen-Shannon Divergence

### [이재윤](https://github.com/pixygear)
- [SEP] pooling

## Citation

- [R-Drop: Regularized Dropout for Neural Networks](https://arxiv.org/abs/2106.14448)
- [AEDA: An Easier Data Augmentation Technique for Text Classification](https://arxiv.org/abs/2108.13230)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
