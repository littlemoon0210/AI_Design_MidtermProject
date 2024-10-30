from datasets import load_dataset

# Load the TinyStories dataset
ds = load_dataset("roneneldan/TinyStories")

# Define the output file path
output_file = "./data/tinystories/TinyStories.txt"

# Open the file and write the content
with open(output_file, "w", encoding="utf-8") as f:
    for example in ds["train"]:  # Using the 'train' set here
        story = example['text']  # Assuming the text field is named 'text'
        f.write(story + "\n\n")  # Add a blank line between stories

print(f"Dataset has been successfully saved to {output_file}")
