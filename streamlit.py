import streamlit as st
import requests
import uuid
import time  # Import time module to track response time

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/chat"

# Initialize session ID if not already present
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())  # Generate a unique session ID

# Set the default language
language = "en"

# Streamlit UI setup
st.title("💬 FastAPI Chatbot with Response Time")
st.write("Ask me anything!")

# Initialize chat history if not already present
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Type your message...")
if user_input:
    #
    # Display user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare request payload
    payload = {
        "message": user_input,
        "session_id": st.session_state["session_id"]
    }

    # Track the start time before making the request
    start_time = time.time()

    # Send request to FastAPI chatbot API
    response = requests.post(API_URL, json=payload)

    # Track the end time after receiving the response
    end_time = time.time()

    # Calculate response time
    response_time = round(end_time - start_time, 2)

    if response.status_code == 200:
        bot_reply = response.json().get("response", "I couldn't understand that.")
    else:
        bot_reply = "Error connecting to chatbot API."

    # Display chatbot response along with response time
    st.session_state["messages"].append({"role": "assistant", "content": f"{bot_reply} \n\n  ⏳ *Latency: {response_time} s* "})
    
    with st.chat_message("assistant"):
        st.markdown(f"{bot_reply} \n\n  ⏳ *Latency: {response_time} s* ")
