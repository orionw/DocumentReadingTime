from flair.datasets import *
from flair.embeddings import *
from flair.data import *
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings
from flair.models.text_regression_model import TextRegressor
from flair.trainers import *
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, ShuffleSplit
import pandas as pd

np.random.seed(31415)

def create_document_embedders(use_rnn: bool = True):
  # load article text
  current_location = os.getcwd()
  df = pd.read_csv(os.path.join("data", "screening_study_with_article_info.csv"), encoding="utf-8")
  text_df = pd.read_csv(os.path.join("data", "text_data.csv"), encoding="utf-8", index_col=0)
  y = df["Reading_Time"]
  df.drop("Reading_Time", axis=1, inplace=True)
  df.drop("Duration_Entire_Survey", axis=1, inplace=True)
  df = pd.get_dummies(df)

  full_train = df.copy(deep=True)
  full_train["Reading_Time"] = y
  full_train = full_train.merge(text_df, on="Survey_Num", how="outer")
  full_train = full_train[["text", "Reading_Time"]]

  sentences = []
  sentences_truncated = []
  for idx in full_train.index:
    data = full_train.iloc[idx]
    sentence = Sentence(data["text"], labels=[Label(value=str(data["Reading_Time"]))], use_tokenizer=True)
    sentences.append(sentence)
    data.text = " ".join(data.text.split(" ")[:300])
    sentence = Sentence(data["text"], labels=[Label(value=str(data["Reading_Time"]))], use_tokenizer=True)
    sentences_truncated.append(sentence)

  ss = ShuffleSplit(n_splits=1, test_size=.3)
  for train, overall_test in ss.split(full_train["text"], full_train["Reading_Time"]):
      # create dataset
      print("The sizes of the train test are", len(train), len(overall_test))
      dev, test = train_test_split(overall_test,  test_size=0.5)
      sent_train, sent_dev, sent_test = [sentences[t] for t in train], [sentences[d] for d in dev], [sentences[e] for e in test]
      # create Dataset objects
      train_dataset = SentenceDataset(sent_train)
      dev_dataset = SentenceDataset(sent_dev)
      test_dataset = SentenceDataset(sent_test)
      # make a Corpus object
      corpus = Corpus(train_dataset, dev_dataset, test_dataset)
      for (name, embeddings) in [("XLNet", XLNetEmbeddings()), ("ELMo", ELMoEmbeddings())]:
          print("Using the model {}".format(name))
          if use_rnn:
            print("Using RNN embeddings")
            document_embeddings = DocumentRNNEmbeddings([embeddings], hidden_size=1024, rnn_layers=1, reproject_words=False, reproject_words_dimension=False, 
                                                      bidirectional=True, dropout=0.4, rnn_type="RNN")
          else:
            print("Using pooling embeddings")
            document_embeddings = DocumentPoolEmbeddings([embeddings], pooling='mean')

          if not os.path.isdir(os.path.join("saved_models", name)):
            os.makedirs(os.path.join("saved_models", name))

          model = TextRegressor(document_embeddings)
          # If this fails on the second model, re-run it with only one of the embedding models (XLNet, ElMO).  I think the tokenization
          # might be off every once it in a while
          trainer = ModelTrainer(model, corpus)
          trainer.train("saved_models/{}/".format(name), learning_rate=0.05, mini_batch_size=16, anneal_factor=0.5, patience=5, max_epochs=100, 
                          embeddings_storage_mode="cuda")
          results = model.evaluate(corpus.test)[0].detailed_results
          print(results)
          model.save(os.path.join("saved_models", name, "{}-final-explicit.pt".format(name)))

  # roBERTa is seperate because the inputs need to be truncated to 512 tokens
  ss = ShuffleSplit(n_splits=1, test_size=.3)
  for train, overall_test in ss.split(full_train["text"], full_train["Reading_Time"]):
      # create dataset
      print("The sizes of the train test are", len(train), len(overall_test))
      dev, test = train_test_split(overall_test,  test_size=0.5)
      sent_train, sent_dev, sent_test = [sentences_truncated[t] for t in train], [sentences_truncated[d] for d in dev], [sentences_truncated[e] for e in test]
      # create Dataset objects
      train_dataset = SentenceDataset(sent_train)
      dev_dataset = SentenceDataset(sent_dev)
      test_dataset = SentenceDataset(sent_test)
      # make a Corpus object
      corpus = Corpus(train_dataset, dev_dataset, test_dataset)
      for (name, embeddings) in [("roBERTa", RoBERTaEmbeddings())]:
          print("Using the embedder {}".format(name))
          if use_rnn:
            document_embeddings = DocumentRNNEmbeddings([embeddings], hidden_size=1024, rnn_layers=1, reproject_words=False, reproject_words_dimension=False, 
                                                      bidirectional=True, dropout=0.4, rnn_type="RNN")
          else:
             document_embeddings = DocumentPoolEmbeddings([embeddings], pooling='mean')

          if not os.path.isdir(os.path.join("saved_models", name)):
            os.makedirs(os.path.join("saved_models", name))
            
          model = TextRegressor(document_embeddings)
          trainer = ModelTrainer(model, corpus)
          trainer.train("saved_models/{}/".format(name), learning_rate=0.05, mini_batch_size=16, anneal_factor=0.5, patience=5, max_epochs=100, 
                          embeddings_storage_mode="cuda")
          results = model.evaluate(corpus.test)[0].detailed_results
          print(results)
          model.save(os.path.join("saved_models", name, "{}-final-explicit.pt".format(name)))


if __name__ == "__main__":
  create_document_embedders(use_rnn=True)  # use True for rnn, False for mean embeddings

