import transformers

# define classifier
classifier = transformers.pipeline("zero-shot-classification", model = "facebook/bart-large-mnli")
# operation
sequence = "I can perform article"
labels = ["writing", "management", "checking"]
classifier(sequence, labels)


