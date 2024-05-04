import numpy as np
import pandas as pd

continent_to_label = {'Asia': 0, "Europe": 1, "North America": 2, "South America": 3, "Africa": 4, "Oceania": 5}

df = pd.read_csv('./balanced_world_place.csv')
df['is_test'] = False
# for k in continent_to_label.keys():
#     k_data = datafile[datafile['continent'] == k]
#     new_idx = datafile.index[datafile[datafile['continent'] == k]].to_list()
#     print(new_idx)
#     np.random.shuffle(new_idx)
#     split_idx = int(len(k_data) * 0.8)
#     for i in range(split_idx):
#         datafile.iloc[new_idx[i]]['is_test'] = False
#     for i in range(split_idx+1, len(k_data)):
#         datafile.iloc[new_idx[i]]['is_test'] = True

def split_continent_data(group):
    # Shuffle the indices of the group
    shuffled_indices = np.random.permutation(group.index)
    # Determine the number of test samples
    test_set_size = int(len(shuffled_indices) * 0.2)
    # Test indices are the last 20% of the shuffled indices
    test_indices = shuffled_indices[-test_set_size:]
    # Set 'is_test' to True for test indices
    group.loc[test_indices, 'is_test'] = True
    return group

# Apply the function to each continent group
new_df = df.groupby('continent', group_keys=True).apply(split_continent_data)

new_df.to_csv('./balanced_world_place_masked.csv')