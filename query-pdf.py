import streamlit as st
#from dotenv import load_dotenv
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA



os.environ['PINECONE_API_KEY'] == st.secrets["PINECONE_API_KEY"]
os.environ['OPENAI_API_KEY'] == st.secrets["OPENAI_API_KEY"]
embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))



def query_vectordb(query,source):
    if source == "CEEW Policy Guidelines":
        index_name = 'ceew-internal-2'
    else:
        index_name = 'ceew-internal-5'
    docsearch = PineconeVectorStore(index_name=index_name, embedding=embedding)
    # Setting main variables
    llm_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=llm_name, temperature=0,openai_api_key=os.getenv("OPENAI_API_KEY"))
    # Build prompt
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible.
    Never answer a question cannot be answered from the context itself. Context Should be the only source of information for the answer.
    Context: {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template)
    qa_chain = RetrievalQA.from_chain_type(llm,retriever=docsearch.as_retriever(), 
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},return_source_documents=True)
    result = qa_chain({"query": query})
        
    return result


st.title("💬 CEEW Bot")
with st.form("qabot"):
    question = st.text_input('Ask your Query',key="question")
    source = st.radio(
    "Select source of your Answer",
    ["CEEW Policy Guidelines", "Published Reports/Content"],
    key="source")
    submitted = st.form_submit_button('Ask')
    if submitted:
       result = query_vectordb(st.session_state.question,st.session_state.source)
       st.write(result['result'])
       doc_list = []
       for i in range(len(result['source_documents'])):
        doc_list.append(result['source_documents'][i].metadata['source'])
       unique_elements = set(doc_list)
        # Convert the set to a list to be able to index elements
       unique_list = list(unique_elements)
       st.write("References")
       for i, element in enumerate(unique_list, start=1):
            st.write(f"{i}. {element}")
