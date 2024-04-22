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
pattern = r'/folders/([a-zA-Z0-9-_]+)'
llm = ChatCohere()
output_parser = StrOutputParser()
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a subject teacher you have create a structured notes based on the input"),
    ("user", "{input}")
])
chain = prompt | llm | output_parser

question_prompt = PromptTemplate(
    input_variables=["notes"],
    template="Generate 2-mark and 5-mark questions based on the following notes:\n\n{notes}",
)

def generate_questions(notes):
    prompt = question_prompt.format(notes=notes)
    response = llm.invoke(prompt)
    
    return response
def clicked():
  st.write("Clicked")
if __name__ == "__main__":
    st.header("Chat With Notes")
    # st.write("Upload or Attach Google Drive Link")
    # st.file_uploader(accept_multiple_files=True, type=["pptx", "pdf", "docx"], label="Upload Document")
    # st.write("or")
    text_input_value = st.text_input(label="Drive Folder Link")
    if st.button("Submit"):
        if text_input_value.startswith("https://drive.google.com/drive/folders/"):
            folderid = re.search(pattern, text_input_value).group(1)
            loader = GoogleDriveLoader(folder_id=folderid, credentials_path="E:\\Workspace\\chat-with-notes\\credentials.json", token_path="E:\\Workspace\\chat-with-notes\\token.json", file_loader_kwargs={"mode": "elements"})
            docs = loader.load()
            content = ""
            for i in range(len(docs)):
                content = content + docs[i].page_content
            response = chain.invoke({"input": content})
            st.write(response)
            if response != None:
              questions = generate_questions(response)
              st.write(questions.content),
              
        else:
            st.write("Invalid Drive link")