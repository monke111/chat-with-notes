import streamlit as st
import re
from langchain_google_community import GoogleDriveLoader
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/drive.metadata.readonly"]

# Regular expression pattern for extracting folder ID from Google Drive folder link
pattern = r'/folders/([a-zA-Z0-9-_]+)'

# Initialize ChatCohere and output parser
llm = ChatCohere()
output_parser = StrOutputParser()

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
  response = llm.invoke(prompt.assemble())
  return response

# Function for chat with cohere
def chat_with_cohere(user_input):
  # Access notes_content from session state
  notes_content = st.session_state.get("notes_content", "")

  # Combine user input and notes_content for ChatCohere prompt
  prompt = ChatPromptTemplate.from_messages([
      ("user", user_input),
      ("system", notes_content)  # Add notes_content to prompt
  ])

  response = llm.invoke({"input": prompt})
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
          credentials_path="E:\\Workspace\\chat-with-notes\\credentials.json",
          token_path="E:\\Workspace\\chat-with-notes\\token.json",
          file_loader_kwargs={"mode": "elements"}
      )
      docs = loader.load()

      # Concatenate the content of all documents
      content = "".join(doc.page_content for doc in docs)

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

  user_input = st.chat_input("Chat with Notes")
  if user_input is not None:
      response= chat_with_cohere(user_input)
      st.write(response)