import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
from PIL import Image

API = st.secrets["API"]

# Set up Streamlit app
image= Image.open("app_banner.png")
st.image(image, use_column_width=True)

# Define app header and footer
# st.markdown(
#     """
#     <div style='padding: 10px; background-color: #EBF5FB;'>
#         <h1 style='text-align: center; color: #283593;'>ChattyPanda</h1>
#     </div>
#     """,
#     unsafe_allow_html=True)

st.markdown(
    """
    <footer style='text-align: center; padding-top: 30px;'>
        Created with ❤️ by Team Menlo Park Boys
    </footer>
    """,
    unsafe_allow_html=True,
)
st.markdown(" **:red[Note :]** :blue[This app is a prototype and Model is trained on limited Data of Emirates Airline] :green[...Thanks for attention. !] ")
# Load and process documents
st.write("Loading and processing documents...")
loader = DirectoryLoader("data", glob="**/*.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Set up question-answering model
st.write("Setting up question-answering model...")
embeddings = OpenAIEmbeddings(openai_api_key=API)
docsearch = Chroma.from_documents(texts, embeddings)
qa = VectorDBQA.from_chain_type(llm=OpenAI(openai_api_key=API), chain_type="map_reduce", vectorstore=docsearch)



# Ask question
st.write("Enter your question below and click 'Ask' to get an answer:")
query = st.text_input("Question: ")
if st.button("Ask"):
    if not query:
        st.warning("Please enter a question.")
    else:
        st.write("Searching for answer...")
        answer = qa.run(query)
        if answer:
            st.success(f"Answer: {answer}")
        else:
            st.error("Sorry, no answer was found.")

# Set page footer
footer = """
<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Made with ❤️: by Menlo Park Boys</p>
<p>View the code on <a href="https://github.com/yourusername/your-repo-name">GitHub</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
