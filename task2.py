import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
import numpy as np

# Load ratings data
ratings = pd.read_csv('ratings.csv')

# Creating a user-item matrix 
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Split the data into training and test sets 
X_data, test_data = train_test_split(user_item_matrix, test_size=0.2, random_state=42)

# Convert training data to matrix
train_matrix = X_data.values
test_matrix = test_data.values

# SVD
svd = TruncatedSVD(n_components=50, random_state=42)
decomposed_matrix = svd.fit_transform(train_matrix)

# Reconstruct the training matrix
reconstructed_matrix = np.dot(decomposed_matrix, svd.components_)

# Compute Mean Squared Error on training data
train_actual = train_matrix[train_matrix.nonzero()].flatten()
train_predicted = reconstructed_matrix[train_matrix.nonzero()].flatten()
train_mse = mean_squared_error(train_actual, train_predicted)
print(f"Mean Squared Error on Training Data: {train_mse}")

# Reconstruct predictions for test data
# Project test data into the same latent space
test_decomposed_matrix = np.dot(test_matrix, svd.components_.T)
test_reconstructed_matrix = np.dot(test_decomposed_matrix, svd.components_)

# Compute Mean Squared Error on test data
test_actual = test_matrix[test_matrix.nonzero()].flatten()
test_predicted = test_reconstructed_matrix[test_matrix.nonzero()].flatten()
test_mse = mean_squared_error(test_actual, test_predicted)
print(f"Mean Squared Error on Test Data: {test_mse}")

# Precision at K function
def precision_at_k(actual, predicted, k=10):
    relevant = 0
    total_recommended = 0

    for user in range(actual.shape[0]):
        # Get the indices of the top K recommended items (sorted by predicted score)
        recommended_items = np.argsort(predicted[user])[::-1][:k]

        # Get the actual rated items for the user (i.e., non-zero values in the user row)
        actual_ratings = np.nonzero(actual[user])[0]

        # Count how many of the top K recommendations are actually rated by the user
        relevant += len(set(recommended_items) & set(actual_ratings))
        total_recommended += k

    # Return precision as the ratio of relevant items in top K to total recommended items
    return relevant / total_recommended

# Compute Precision at K
precision_k = precision_at_k(test_matrix, reconstructed_matrix, k=10)
print(f"Precision at 10: {precision_k}")
