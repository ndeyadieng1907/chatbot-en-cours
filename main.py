import streamlit as st
from openai import OpenAI
from PIL import Image
import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import asyncio
load_dotenv()

st.set_page_config(page_title="ANSD'S bot", page_icon=":ansd-sn:",layout="wide")

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("sk-proj-mS9M2ILLVhujWjwccWdTT3BlbkFJo6dm8QOPGM41PfsfNATQ"),
)



# st.title("Bienvenue Un centre d'excellence dans un Système statistique national fort


st.markdown("<h2 style='text-align: center; '>l'assistant vocal de l'ansd <br>   ", unsafe_allow_html=True)
st.markdown('<style>div.block-container{padding-top:2rem;}</style>',unsafe_allow_html=True)

col1,col2 = st.columns(2)
st.write("""Découvrez ANSD'S bot, notre chatbot alimenté par l'intelligence artificielle vous offre un aperçu de notre solution""")

 
def load_db(file, chain_type, k):
    async def load_and_process():
        # load documents 
        loader = PyPDFLoader(file)
        documents = loader.load()
        # split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)
        # define embedding
        embeddings = OpenAIEmbeddings(openai_api_key = os.environ.get("sk-proj-mS9M2ILLVhujWjwccWdTT3BlbkFJo6dm8QOPGM41PfsfNATQ"))
        # create vector database from data
        db = DocArrayInMemorySearch.from_documents(docs, embeddings)
        # define retriever
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
        # create a chatbot chain. Memory is managed externally.
        qa = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=os.environ.get("OPENAI_API_KEY")),
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True,
            return_generated_question=True,
        )
        return qa

    return asyncio.run(load_and_process()) 

async def chat_with_bot(cb, query):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, cb, {"question": query, "chat_history": []})

def main():
    st.title("  ANSD's bot ✨")
    
    col1,col2 = st.columns(2)
    with col1:
        st.write("Je suis votre assistant dédié a l'ANSD. N'hésitez pas à me poser des questions sur les informations statistique du Senegal. Par exemple, vous pouvez me demander des informations sur l'espérance de vie à la naissance pour les hommes et les femmes , la répartition des décès par sexe et par âge . Je suis là pour vous fournir des réponses précises et utiles .")
    with col2:
        original_image = Image.open("image.png")
        st.image(original_image)

    # Load the database and chatbot
    cb = load_db("MORTALITE-Rapport-Provisoire-RGPH5_juillet2024.pdf", "stuff", 4)

    query = st.text_input("Poser une question:")
    if st.button("Demander"):
        result = asyncio.run(chat_with_bot(cb, query))
        response = result["answer"]
        st.write("ChatBot:", response)

if __name__ == '__main__':
    main()
# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data


# splitting data in chunks
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)

    # if you want to use a specific directory for chromadb
    # vector_store = Chroma.from_documents(chunks, embeddings, persist_directory='./mychroma_db')
    return vector_store


def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.run(q)
    return answer


# calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
    return total_tokens, total_tokens / 1000 * 0.0004


# clear the chat history from streamlit session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


if __name__ == "__main__":
    import os

    # loading the OpenAI api key from .env
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    st.subheader('LLM Question-Answering Application 🤖')
    with st.sidebar:
        # text_input for the OpenAI API key (alternative to python-dotenv and .env)
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        # file uploader widget
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])

        # chunk size number widget
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)

        # k number input widget
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)

        # add data button widget
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data: # if the user browsed a file
            with st.spinner('Reading, chunking and embedding file ...'):

                # writing the file from RAM to the current directory on disk
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')

                # creating the embeddings and returning the Chroma vector store
                vector_store = create_embeddings(chunks)

                # saving the vector store in the streamlit session state (to be persistent between reruns)
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully.')

    # user's question text input widget
    q = st.text_input('Ask a question about the content of your file:')
    if q: # if the user entered a question and hit enter
        if 'vs' in st.session_state: # if there's the vector store (user uploaded, split and embedded a file)
            vector_store = st.session_state.vs
            answer = ask_and_get_answer(vector_store, q, k)

            # text area widget for the LLM answer
            st.write(f"Chatbot : {answer}")

            # st.divider()

            # # if there's no chat history in the session state, create it
            # if 'history' not in st.session_state:
            #     st.session_state.history = ''

            # # the current question and answer
            # value = f'Q: {q} \nA: {answer}'

            # st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            # h = st.session_state.history

            # # text area widget for the chat history
            # st.text_area(label='Chat History', value=h, key='history', height=400)

# run the app: streamlit run ./chat_with_documents.py
