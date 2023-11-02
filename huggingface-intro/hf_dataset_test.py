from datasets import load_dataset

conll03 = load_dataset("conll2003")
features = conll03["train"].features[f"ner_tags"].feature.names
print(features)

for i in range(5):
    print(f"Sentence {i}")
    sentence = conll03["train"][i]
    print(sentence)
    print("---")