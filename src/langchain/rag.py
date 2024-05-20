from langchain_community.llms import Bedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
import boto3


myData = [
    "The weather is nice today.",
    "Last night's game ended in a tie.",
    "Don likes to eat pizza.",
    "Don likes to eat pasta.",
]

question = "What are Don's favorite foods?"

AWS_REGION = "us-west-2"
bedrock = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)
bedrock_embaddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)
model = Bedrock(model_id="amazon.titan-text-express-v1", client=bedrock)

vector_store = FAISS.from_texts(myData, bedrock_embaddings)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

results = retriever.get_relevant_documents(question)

result_strings = []

for result in results:
    result_strings.append(result.page_content)
    
    
template = ChatPromptTemplate.from_messages(
    [
        
            ("system", "Answer the users question based on following context {context}" ),
            ("user", "{input}")

    ]
) 

chain = template|model 

response = chain.invoke({
    "input": question,
    "context": result_strings
})

print(response)

