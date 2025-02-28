import numpy as np

def positional_encoding(values, num_frequencies):
    encoded = []
    for value in values:
        for i in range(num_frequencies):
            encoded.append(np.sin((2 ** i) * value))
            encoded.append(np.cos((2 ** i) * value))
    return np.array(encoded)

# Step 1: Encode x values
x_values = np.array([1.0, 2.0, 3.0])  # Example x coordinates
x_encoded = np.array([positional_encoding([x], 5) for x in x_values])

# Step 2: Encode y values using encoded x values
y_values = np.array([4.0, 5.0, 6.0])  # Example y coordinates
y_encoded = []
for y, x_enc in zip(y_values, x_encoded):
    y_encoded.append(positional_encoding([y] + list(x_enc), 5))
y_encoded = np.array(y_encoded)

# Step 3: Encode z values using encoded x and y values
z_values = np.array([7.0, 8.0, 9.0])  # Example z coordinates
z_encoded = []
for z, y_enc, x_enc in zip(z_values, y_encoded, x_encoded):
    z_encoded.append(positional_encoding([z] + list(y_enc) + list(x_enc), 5))
z_encoded = np.array(z_encoded)

print("Encoded x values:\n", x_encoded)
print("Encoded y values:\n", y_encoded)
print("Encoded z values:\n", z_encoded)
