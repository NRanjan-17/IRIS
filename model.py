from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import json
import os
from datetime import datetime

DB_FAISS_PATH = 'vectorstore/db_faiss'
CONVERSATION_LOG_PATH = 'conversation_logs'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt})
    return qa_chain

# Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGUF",
        model_file="llama-2-7b-chat.Q5_K_M.gguf",
        model_type="llama",
        max_new_tokens=1096,
        repetition_penalty=1.13,
        temperature=0.5,
    )
    return llm

# QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

# Output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

# Function to save the conversation
def save_conversation(conversation):
    if not os.path.exists(CONVERSATION_LOG_PATH):
        os.makedirs(CONVERSATION_LOG_PATH)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(CONVERSATION_LOG_PATH, f'conversation_log_{timestamp}.json')
    
    with open(file_path, 'w') as f:  # Use 'w' to create a new file for each session
        json.dump(conversation, f, indent=4)

# Chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Iris: Your Personal Health Companion. What is your query?"
    await msg.update()

    # Set default prompts
    default_prompts = [
        "What are the symptoms of diabetes?",
        "How can I lower my blood pressure?",
        "What should I eat to stay healthy?",
        "Tell me about the side effects of aspirin."
    ]
    
    cl.user_session.set("chain", chain)
    cl.user_session.set("conversation", [])
    cl.user_session.set("default_prompts", default_prompts)

    # Send default prompts as separate messages
    await cl.Message(content="Here are some example questions you can ask:").send()
    for prompt in default_prompts:
        await cl.Message(content=prompt).send()

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    conversation = cl.user_session.get("conversation")
    
    # Log user message
    conversation.append({"role": "user", "message": message.content})

    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.ainvoke(message.content, callbacks=[cb])  # Updated method
    answer = res["result"]
    # sources = res["source_documents"]

    # if sources:
    #     answer += f"\nSources:" + str(sources)
    # else:
    #     answer += "\nNo sources found"

    # Log bot response
    conversation.append({"role": "bot", "message": answer})
    
    # Save conversation
    save_conversation(conversation)

    await cl.Message(content=answer).send()
