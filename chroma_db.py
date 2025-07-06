from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
#Uses LangChain's PyPDFLoader to load PDFs.
#RecursiveCharacterTextSplitter breaks large text into chunks (with overlap).
#Uses Google's embedding model for high-quality vectorization.  



# Load and process PDF documents
pdf_paths = ["data/mmm1.pdf", "data/mmm2.pdf","data/mmm3.pdf","data/mmm4.pdf","data/mmm5.pdf"] 
documents = []
for path in pdf_paths:
    loader = PyPDFLoader(path)
    documents.extend(loader.load())


# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Initialize Google embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")  

# Define the directory to persist the vector store
persist_directory = "./chroma_db"

# Create and persist the vector store
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=persist_directory
)
vectorstore.persist()

#Ce script charge des fichiers PDF, les découpe en morceaux, génère des embeddings
# (vecteurs) avec un modèle de Google, puis stocke tout ça dans Chroma pour les retrouver
# plus tard dans une recherche sémantique.