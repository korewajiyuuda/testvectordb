# import
from langchain_chroma import Chroma # type: ignore
from langchain_community.document_loaders import TextLoader # type: ignore
from langchain_community.embeddings.sentence_transformer import ( # type: ignore
    SentenceTransformerEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter # type: ignore
from transformers import AutoTokenizer # type: ignore



# load the document and split it into chunks
loader = TextLoader("./Article-1.txt")
documents = loader.load()

# split it into chunks
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(AutoTokenizer.from_pretrained('distilroberta-base'), chunk_size=150, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# print(embedding_function.max_seq_length)

# load it into Chroma
db = Chroma.from_documents(docs, embedding_function)

# # query it
query = "How do I remove spyware/adware from my computer?"
docs = db.similarity_search(query)

# # print results
print(docs[0].page_content)