import numpy as np

# Replace this with your numpy array
natural_numbers_array = np.array([2, 4, 6, 8, 10, 6, 8, 12, 14])

# Find indices of elements that are 6 or 8
indices_6_or_8 = np.where((natural_numbers_array == 6) | (natural_numbers_array == 8))[0]

# Select elements that are 6 or 8 using the indices
selected_numbers = natural_numbers_array[indices_6_or_8]

print("Selected numbers:", selected_numbers)
