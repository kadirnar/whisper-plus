import lancedb
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.vectorstores import LanceDB
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI


# Configuration and Constants
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL_NAME = 'TheBloke/Mistral-7B-v0.1-GGUF'
LLM_MODEL_FILE = 'mistral-7b-v0.1.Q4_K_M.gguf'
LLM_MODEL_TYPE = "mistral"
TEXT_FILE_PATH = "/content/moe_blog.text"
DATABASE_PATH = '/tmp/lancedb'

class ChatWithVideo:
    @staticmethod
    def load_llm_model():
        print("Mistral model started downloading")
        return CTransformers(model=LLM_MODEL_NAME, model_file=LLM_MODEL_FILE, model_type=LLM_MODEL_TYPE)

    @staticmethod
    def load_text_file(file_path):
        loader = TextLoader(file_path)
        print("Transcript file loaded")
        return loader.load()

    @staticmethod
    def setup_database():
        return lancedb.connect(DATABASE_PATH)

    # embedding model
    @staticmethod
    def prepare_embeddings(model_name):
        return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})

    @staticmethod
    def prepare_documents(docs):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        return text_splitter.split_documents(docs)

    @staticmethod
    def run_query(query):
        docs = ChatWithVideo.load_text_file(TEXT_FILE_PATH)
        documents = ChatWithVideo.prepare_documents(docs)

        embeddings = ChatWithVideo.prepare_embeddings(MODEL_NAME)
        db = ChatWithVideo.setup_database()

        table = db.create_table("pandas_docs", data=[
            {"vector": embeddings.embed_query("Hello World"), "text": "Hello World", "id": "1"}
        ], mode="overwrite")
        docsearch = LanceDB.from_documents(documents, embeddings, connection=table)

        llm = ChatWithVideo.load_llm_model()
        #llm =  ChatOpenAI(openai_api_key="sk-") use openai model directly


        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.
        Always say "thanks for asking!" at the end of the answer.
        {context}
        Question: {question}
        Helpful Answer:"""

        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
        print("prompt loaded")
        qa = RetrievalQA.from_chain_type(llm, chain_type='stuff', retriever=docsearch.as_retriever(), chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
        return qa.run(query)
