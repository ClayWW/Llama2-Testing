# Importing necessary modules and classes
from transformers import AutoTokenizer  # Import AutoTokenizer for tokenizing input text
import transformers  # Import the transformers library
import torch  # Import the PyTorch library

# Define the model to be used
model = "meta-llama/LLama-2-7b-chat-hf"  # Specify the Llama2 model available on Hugging Face

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model)  # Initialize the tokenizer specific to the Llama2 model

# Set up the text generation pipeline
pipeline = transformers.pipeline(
    "text-generation",  # Specify the task as text generation
    model=model,  # Use the specified Llama2 model
    torch_dtype=torch.float16,  # Set the data type for PyTorch tensors to float16 (reduced precision for efficiency)
    device_map="auto",  # Automatically use available device (CPU or GPU) for model computation
)

# Generate sequences based on the input prompt
sequences = pipeline(
    'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    do_sample=True,  # Enable random sampling for generating text
    top_k=10,  # Limit the number of highest probability vocabulary tokens to consider for each step
    num_return_sequences=1,  # Return only one sequence
    eos_token_id=tokenizer.eos_token_id,  # Specify the end-of-sequence token ID for stopping generation
    max_length=200,  # Maximum length of the generated sequences
)

# Print the generated sequences
for seq in sequences:
    print(f"Result: {seq['generated_text']}")  # Format and print each generated sequence
