import lancedb
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import LanceDB
from langchain.prompts import PromptTemplate


class ChatWithVideo:
    def __init__(self,input_file ,llm_model_name, llm_model_file, llm_model_type, embedding_model_name):
        self.input_file = input_file
        self.llm_model_name = llm_model_name
        self.llm_model_file = llm_model_file
        self.llm_model_type = llm_model_type
        self.embedding_model_name = embedding_model_name

    def load_llm_model(self):
        try:
            print(f"Starting to download the {self.llm_model_name} model...")
            llm_model = CTransformers(
                model=self.llm_model_name, model_file=self.llm_model_file, model_type=self.llm_model_type)
            print(f"{self.llm_model_name} model successfully loaded.")
            return llm_model
        except Exception as e:
            print(f"Error loading the {self.llm_model_name} model: {e}")
            return None


    def load_text_file(self):
        try:
            print(f"Loading transcript file from {self.input_file}...")
            loader = TextLoader(self.input_file)
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
            db = lancedb.connect('/tmp/lancedb')
            print("Database setup complete.")
            return db
        except Exception as e:
            print(f"Error setting up the database: {e}")
            return None

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

    def run_query(self, query):
        if not query:
            print("No query provided.")
            return "No query provided."

        print(f"Running query: {query}")
        docs = self.load_text_file()
        if not docs:
            return "Failed to load documents."

        documents = self.prepare_documents(docs)
        if not documents:
            return "Failed to prepare documents."

        embeddings = self.prepare_embeddings(self.embedding_model_name)
        if not embeddings:
            return "Failed to prepare embeddings."

        db = self.setup_database()
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

            llm = self.load_llm_model()
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
            print("Prompt loaded")
            qa = RetrievalQA.from_chain_type(
                llm,
                chain_type='stuff',
                retriever=docsearch.as_retriever(),
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
            print("Query processed successfully.")

            result = qa.run(query)
            print("Result of the query:", result)
            return result
        except Exception as e:
            print(f"Error running query: {e}")
            return f"Error: {e}"






# input_file = '/content/moe_blog.text'
# llm_model_name = 'TheBloke/Mistral-7B-v0.1-GGUF'
# llm_model_file = 'mistral-7b-v0.1.Q4_K_M.gguf'
# llm_model_type = "mistral"
# embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
# chat = ChatWithVideo(input_file,llm_model_name, llm_model_file, llm_model_type, embedding_model_name)
# query = "what is mistral ?"
# response = chat.run_query(query)
# print(response)
