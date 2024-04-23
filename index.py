import streamlit as st
import re
from langchain_google_community import GoogleDriveLoader
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_cohere.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/drive.metadata.readonly"]

# Regular expression pattern for extracting folder ID from Google Drive folder link
pattern = r'/folders/([a-zA-Z0-9-_]+)'

# Initialize ChatCohere and output parser
llm = ChatCohere()
output_parser = StrOutputParser()
embeddings = CohereEmbeddings()
text_splitter = RecursiveCharacterTextSplitter()
# Define prompts and chains
prompt = ChatPromptTemplate.from_messages([
  ("system", "You are a subject teacher. You have to create structured notes based on the input."),
  ("user", "{input}")
])

question_prompt = PromptTemplate(
  input_variables=["notes"],
  template="Generate 2-mark and 5-mark questions based on the following notes:\n\n{notes}",
)

chain = prompt | llm | output_parser

# Function to generate questions
def generate_questions(notes):
  prompt = question_prompt.format(notes=notes)
  response = llm.invoke(prompt)
  return response


def get_vectorstore(content):
  document = text_splitter.split_text(content)
  vector = FAISS.from_texts(document, embeddings)
  return vector

def get_context_retriever_chain(vector_store):
  retriever = vector_store.as_retriever()
  prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the current question. " +
          "Don't leave out any relevant keywords. Only return the query and no other text.",)
    ])
  
  retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
  return retriever_chain

def get_conversational_rag_chain(retriever_chain):
  prompt = ChatPromptTemplate.from_messages([
      ("system", "You are a personal tutor for students attending a workshop. You impersonate the workshop instructor. " +
          "Answer the user's questions based on the below context. " +
          "Whenever it makes sense, provide links to pages that contain more information about the topic from the given context. " +
          "Format your messages in markdown format.\n\n" +
          "Context:\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
  stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
  return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# Function for chat with cohere
def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response_stream = conversation_rag_chain.stream({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })

    response = ""
    for chunk in response_stream:
        content = chunk.get("answer", "")
        response += content

    return response

# Main function
if __name__ == "__main__":
  st.sidebar.header("AI Notes Assistant")
  text_input_value = st.sidebar.text_input(label="Drive Folder Link")

  if st.sidebar.button("Submit"):
    if text_input_value.startswith("https://drive.google.com/drive/folders/"):
      # Extract folder ID from the input link
      folderid = re.search(pattern, text_input_value).group(1)

      # Load documents from the specified Google Drive folder
      loader = GoogleDriveLoader(
          folder_id=folderid,
          credentials_path="C:\\Harish\\workspace\\chat-with-notes\\credentials.json",
          token_path="C:\\Harish\\workspace\\chat-with-notes\\token.json",
          file_loader_kwargs={"mode": "elements"}
      )
      docs = loader.load()

      # Concatenate the content of all documents
      content = "".join(doc.page_content for doc in docs)
      if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore(content)
      # Generate structured notes
      response = chain.invoke({"input": content})
      

      if response is not None:
        # Generate questions based on the notes
        questions = generate_questions(response)

        # Store notes content and questions content in session state
        st.session_state.notes_content = response
        st.session_state.questions_content = questions.content
      else:
        st.write("Generating Notes")

  # Display notes content, questions content, and chat input
  if "questions_content" in st.session_state:
    st.container(border=True).write(st.session_state.notes_content)
    st.container(border=True).write(st.session_state.questions_content)

  if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am your AI workshop Instructor. Ask me any doubts related to the workshop."),
    ]

  for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

  user_query = st.chat_input("Type your message here...")
  if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    response = get_response(user_query)
    st.session_state.chat_history.append(AIMessage(content=response))
    with st.chat_message("AI"):
        st.write(response)