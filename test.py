import numpy as np

# Initialiser une array vide (2D)
empty_array = np.empty((0, 4))
print(empty_array)

# Array à ajouter
new_array = np.array([[1, 2, 3, 4]])

# Ajouter new_array à empty_array
result = np.vstack((empty_array, new_array))
result = np.vstack((result, new_array))

print(result)