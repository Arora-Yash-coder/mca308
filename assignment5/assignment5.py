import numpy as np
from scipy.spatial.distance import euclidean, cityblock, hamming, chebyshev, minkowski
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity

customer_A = [4, 5, 2, 3, 4]
customer_B = [5, 3, 2, 4, 5]
customer_A_binary = [1, 0, 1, 1, 0, 1]
customer_B_binary = [1, 1, 1, 0, 0, 1]

euclidean_distance = euclidean(customer_A, customer_B)
manhattan_distance = cityblock(customer_A, customer_B)
cosine_sim = cosine_similarity([customer_A], [customer_B])[0][0]
hamming_distance = hamming(customer_A_binary, customer_B_binary) * len(customer_A_binary)
jaccard_sim = jaccard_score(customer_A_binary, customer_B_binary)

user1 = [5, 3, 4, 4, 2]
user2 = [4, 2, 5, 4, 3]

chebyshev_distance = chebyshev(user1, user2)
minkowski_distance = minkowski(user1, user2, 3)

print("Euclidean Distance:", euclidean_distance)
print("Manhattan Distance:", manhattan_distance)
print("Cosine Similarity:", cosine_sim)
print("Hamming Distance:", hamming_distance)
print("Jaccard Similarity:", jaccard_sim)
print("Chebyshev Distance:", chebyshev_distance)
print("Minkowski Distance (p=3):", minkowski_distance)
