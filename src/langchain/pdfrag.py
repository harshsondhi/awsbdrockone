from langchain_community.llms import Bedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import boto3


AWS_REGION = "us-west-2"
bedrock = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)
model = Bedrock(model_id="amazon.titan-text-express-v1", client=bedrock)
bedrock_embaddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

question = "What theme does gone with the wind explore?"

loader = PyPDFLoader("books.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(separators=[". \n"], chunk_size=200)
splited_docs = splitter.split_documents(docs)

vector_store = FAISS.from_documents(splited_docs, bedrock_embaddings)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

results = retriever.get_relevant_documents(question)

results_strings = []

for result in results:
    results_strings.append(result.page_content)
    
tempelate= ChatPromptTemplate.from_messages(
    [
        ("system","Answer the users question based on the following context: {context}"),
        ("user","{input}"),
    ]
) 

cahain = tempelate|model
response = cahain.invoke({
    "input": question,
    "context": results_strings
})

print(response)