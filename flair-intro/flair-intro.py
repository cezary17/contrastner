from flair.datasets import CONLL_03
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

# 1. get the corpus
corpus = CONLL_03()
print(corpus)

for i in range(5):
    print(f"Sentence {i}")
    sentence = corpus.train[i]
    print(sentence.to_tagged_string("ner"))
    print(type(sentence))
    labels = sentence.get_labels("ner")
    for label in labels:
        print(label)
        print(type(label))
        print(label.value)
        print(label.score)
        print(label.unlabeled_identifier)
        print(type(label.unlabeled_identifier))
    print("---")

# # 2. what label do we want to predict?
# label_type = 'ner'
#
# # 3. make the label dictionary from the corpus
# label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=False)
# print(label_dict)
#
# # 4. initialize fine-tuneable transformer embeddings WITH document context
# embeddings = TransformerWordEmbeddings(
#     model='xlm-roberta-large',
#     layers="-1",
#     subtoken_pooling="first",
#     fine_tune=True,
#     use_context=True,
# )
#
# # 5. initialize bare-bones sequence tagger (no CRF, no RNN, no reprojection)
# tagger = SequenceTagger(
#     hidden_size=256,
#     embeddings=embeddings,
#     tag_dictionary=label_dict,
#     tag_type='ner',
#     use_crf=False,
#     use_rnn=False,
#     reproject_embeddings=False,
# )
#
# # 6. initialize trainer
# trainer = ModelTrainer(tagger, corpus)
#
# # 7. run fine-tuning
# trainer.fine_tune(
#     'resources/taggers/sota-ner-flert',
#     learning_rate=5.0e-6,
#     mini_batch_size=4,
#     # mini_batch_chunk_size=1,  # remove this parameter to speed up computation if you have a big GPU
#     max_epochs=2,
# )
