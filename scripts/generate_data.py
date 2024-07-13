import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 1000

profile_age = np.random.randint(1, 3650, n_samples)
number_of_friends = np.random.randint(0, 5000, n_samples)
profile_completeness = np.random.uniform(0, 100, n_samples)
number_of_posts = np.random.randint(0, 5000, n_samples)
profile_picture = np.random.choice([0, 1], n_samples)
about_me_length = np.random.randint(0, 2000, n_samples)
messages_sent = np.random.randint(0, 10000, n_samples)
account_verified = np.random.choice([0, 1], n_samples)

labels = (
    (number_of_friends > 1000) &
    (profile_completeness > 70) &
    (number_of_posts > 100) &
    (profile_picture == 1) &
    (about_me_length > 100) &
    (messages_sent > 1000) &
    (account_verified == 1)
).astype(int)

data = pd.DataFrame({
    'profile_age': profile_age,
    'number_of_friends': number_of_friends,
    'profile_completeness': profile_completeness,
    'number_of_posts': number_of_posts,
    'profile_picture': profile_picture,
    'about_me_length': about_me_length,
    'messages_sent': messages_sent,
    'account_verified': account_verified,
    'label': labels
})

data.to_csv('../data/profile_data.csv', index=False)
