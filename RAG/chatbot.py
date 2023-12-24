import lancedb
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import LanceDB

# Configuration and Constants
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL_NAME = 'TheBloke/Mistral-7B-v0.1-GGUF'
LLM_MODEL_FILE = 'mistral-7b-v0.1.Q4_K_M.gguf'
LLM_MODEL_TYPE = "mistral"
TEXT_FILE_PATH = "transcript.text"
DATABASE_PATH = '/tmp/lancedb'


class ChatWithVideo:

    @staticmethod
    def load_llm_model():
        try:
            print("Starting to download the Mistral model...")
            llm_model = CTransformers(
                model=LLM_MODEL_NAME, model_file=LLM_MODEL_FILE, model_type=LLM_MODEL_TYPE)
            print("Mistral model successfully loaded.")
            return llm_model
        except Exception as e:
            print(f"Error loading the Mistral model: {e}")
            return None

    @staticmethod
    def load_text_file(file_path):
        try:
            print(f"Loading transcript file from {file_path}...")
            loader = TextLoader(file_path)
            docs = loader.load()
            print("Transcript file successfully loaded.")
            return docs
        except Exception as e:
            print(f"Error loading text file: {e}")
            return None

    @staticmethod
    def setup_database():
        try:
            print("Setting up the database...")
            db = lancedb.connect(DATABASE_PATH)
            print("Database setup complete.")
            return db
        except Exception as e:
            print(f"Error setting up the database: {e}")
            return None

    # embedding model
    @staticmethod
    def prepare_embeddings(model_name):
        try:
            print(f"Preparing embeddings with model: {model_name}...")
            embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})
            print("Embeddings prepared successfully.")
            return embeddings
        except Exception as e:
            print(f"Error preparing embeddings: {e}")
            return None

    @staticmethod
    def prepare_documents(docs):
        if not docs:
            print("No documents provided for preparation.")
            return None
        try:
            print("Preparing documents...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
            documents = text_splitter.split_documents(docs)
            print("Documents prepared successfully.")
            return documents
        except Exception as e:
            print(f"Error preparing documents: {e}")
            return None

    @staticmethod
    def run_query(query):
        if not query:
            print("No query provided.")
            return "No query provided."

        print(f"Running query: {query}")
        docs = ChatWithVideo.load_text_file(TEXT_FILE_PATH)
        if not docs:
            return "Failed to load documents."

        documents = ChatWithVideo.prepare_documents(docs)
        if not documents:
            return "Failed to prepare documents."

        embeddings = ChatWithVideo.prepare_embeddings(MODEL_NAME)
        if not embeddings:
            return "Failed to prepare embeddings."

        db = ChatWithVideo.setup_database()
        if not db:
            return "Failed to setup database."

        try:
            table = db.create_table(
                "pandas_docs",
                data=[{
                    "vector": embeddings.embed_query("Hello World"),
                    "text": "Hello World",
                    "id": "1"
                }],
                mode="overwrite")
            docsearch = LanceDB.from_documents(documents, embeddings, connection=table)

            llm = ChatWithVideo.load_llm_model()

            if not llm:
                return "Failed to load LLM model."

            template = """Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Use three sentences maximum and keep the answer as concise as possible.
            Always say "thanks for asking!" at the end of the answer.
            {context}
            Question: {question}
            Helpful Answer:"""

            QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
            print("prompt loaded")
            qa = RetrievalQA.from_chain_type(
                llm,
                chain_type='stuff',
                retriever=docsearch.as_retriever(),
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
            print("Query processed successfully.")
            return qa.run(query)
        except Exception as e:
            print(f"Error running query: {e}")
            return f"Error: {e}"
