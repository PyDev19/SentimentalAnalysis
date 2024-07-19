import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from tqdm import tqdm

data = pd.read_csv('data/preprocessed_data.csv')
X = pd.DataFrame([ids for ids in tqdm(data['input_ids'], desc='Evaluating input_ids')])
y = data['label']

print('Original dataset shape %s' % Counter(y))

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(pd.DataFrame(X), y)

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)

print('Resampled dataset shape %s' % Counter(y_resampled))

data_resampled = pd.DataFrame({'input_ids': X_resampled[0], 'label': y_resampled})

data_resampled.to_csv('data/balanced_preprocessed_data.csv', index=False)
