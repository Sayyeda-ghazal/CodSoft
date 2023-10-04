import easyocr
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Path to the image containing handwritten-like text
img_path = '/content/drive/MyDrive/ghazal/programming/Artificial Intelligence/machine learning/CodSoft/Handwritten-like testing/img1.jpg'

# Read text from the image using EasyOCR
reader = easyocr.Reader(['en'], gpu=False)
result = reader.readtext(img_path)

# Extract text from EasyOCR result
extracted_text = result[0][-2]

# Preprocess the text data
chars = sorted(list(set(extracted_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
n_chars = len(extracted_text)
n_vocab = len(chars)

# Create input and output sequences for the LSTM model
seq_length = 21  # Length of input sequences
dataX = []
dataY = []
for i in range(0, n_chars - seq_length):
    seq_in = extracted_text[i:i + seq_length]
    seq_out = extracted_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])

n_patterns = len(dataX)
X = np.reshape(dataX, (n_patterns, seq_length, 1))
X = X / float(n_vocab)
y = np.array(dataY)

# Convert data to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)

# Define the LSTM model using PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

# Instantiate the model, define loss function and optimizer
input_size = 1
hidden_size = 128
output_size = len(chars)  # Number of unique characters
model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Now you can use the trained model for text generation or other tasks
# Assuming you have already trained the model and have the trained model object

# Set the seed text to start the generation
seed_text = "Qvwmous to Hk"
seed_int = [char_to_int[char] for char in seed_text]

# Generate new characters
generated_text = seed_text
with torch.no_grad():
    for _ in range(100):  # Adjust the number of characters to generate as needed
        # Prepare the input sequence
        input_seq = torch.FloatTensor(seed_int[-seq_length:]).view(1, -1, 1) / float(n_vocab)

        # Predict the next character
        predicted_char_probs = model(input_seq)

        # Reshape the output and apply argmax
        predicted_char_probs = predicted_char_probs.view(-1)  # Reshape to 1D tensor
        next_char_idx = torch.argmax(predicted_char_probs).item()

        # Convert the index to a character
        next_char = int_to_char[next_char_idx]

        # Add the predicted character to the generated text
        generated_text += next_char

        # Update the seed text for the next prediction
        seed_int.append(next_char_idx)

print("Generated Text:")
print(generated_text)
from PIL import Image, ImageDraw, ImageFont


# Generated text
generated_text = "Your generated text here"

# Set image size and font properties
image_width, image_height = 800, 200
font_size = 36

# Create a blank image with white background
image = Image.new('RGB', (image_width, image_height), color='white')

# Use a truetype font (handwritten-like font)
font_path = "/content/drive/MyDrive/ghazal/programming/Artificial Intelligence/machine learning/CodSoft/Handwritten-like testing/Roboto-Thin.ttf"
try:
    font = ImageFont.truetype(font_path, font_size)
except IOError:
    font = ImageFont.load_default()  # Use default font if specified font is not found

# Draw the text on the image
draw = ImageDraw.Draw(image)

# Get the bounding box of the text
text_bbox = draw.textbbox((0, 0), generated_text, font=font)

# Calculate the position to center the text in the image
text_position = ((image_width - (text_bbox[2] - text_bbox[0])) // 2, (image_height - (text_bbox[3] - text_bbox[1])) // 2)

# Add text to the image
draw.text(text_position, generated_text, font=font, fill='black')

# Save or display the image
image.save("generated_text_image.png")
image.show()
