from langchain_community.vectorstores.utils import DistanceStrategy # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter # type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings # type: ignore
from transformers.agents import Tool, HfEngine, ReactJsonAgent # type: ignore
from langchain_community.document_loaders import TextLoader # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore
from langchain.docstore.document import Document # type: ignore
from transformers import AutoTokenizer # type: ignore
from tqdm import tqdm # type: ignore
import datasets, logging # type: ignore
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


loader = TextLoader("../Article-1.txt")
documents = loader.load()

logger.info(f"Loaded {len(documents)} documents from the knowledge base")

tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer,
    chunk_size=150,
    chunk_overlap=0,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

logger.info("Splitting documents...")
docs_processed = []
unique_texts = {}
for doc in tqdm(documents):
    new_docs = text_splitter.split_documents([doc])
    for new_doc in new_docs:
        if new_doc.page_content not in unique_texts:
            unique_texts[new_doc.page_content] = True
            docs_processed.append(new_doc)

logger.info(f"Processed {len(docs_processed)} unique document chunks")

logger.info("Initializing embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")

logger.info("Creating vector database...")
vectordb = FAISS.from_documents(
    documents=docs_processed,
    embedding=embedding_model,
    distance_strategy=DistanceStrategy.COSINE,
)

logger.info("Vector database created successfully")

class RetrieverTool(Tool):
    name = "retriever"
    description = "Using semantic similarity, retrieves some documents from the knowledge base that have the closest embeddings to the input query."
    inputs = {
        "query": {
            "type": "text",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "text"

    def __init__(self, vectordb, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.vectordb.similarity_search(
            query,
            k=7,
        )

        return "\nRetrieved documents:\n" + "".join(
            [f"===== Document {str(i)} =====\n" + doc.page_content for i, doc in enumerate(docs)]
        )

# Create an instance of the RetrieverTool
retriever_tool = RetrieverTool(vectordb)

llm_engine = HfEngine("meta-llama/Meta-Llama-3-8B-Instruct")

agent = ReactJsonAgent(tools=[retriever_tool], llm_engine=llm_engine, max_iterations=3, verbose=2)

def run_agentic_rag(question: str) -> str:
    enhanced_question = f"""Using the information contained in your knowledge base, which you can access with the 'retriever' tool,
give a comprehensive answer to the question below.
Respond only to the question asked, response should be concise and relevant to the question.
If you cannot find information, do not give up and try calling your retriever again with different arguments!
Make sure to have covered the question completely by calling the retriever tool several times with semantically different queries.
Your queries should not be questions but affirmative form sentences: e.g. rather than "How do I load a model from the Hub in bf16?", query should be "load a model from the Hub bf16 weights".

Question:
{question}"""

    return agent.run(enhanced_question)

question = "How can I remove spyware/adware?"
answer = run_agentic_rag(question)
print(f"Question: {question}")
print(f"Answer: {answer}")