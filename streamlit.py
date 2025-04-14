import streamlit as st
import requests
import uuid
import time
import html

API_URL = "http://127.0.0.1:8000/chat"

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.set_page_config(page_title="Chatbot", page_icon="💬")
st.title("💬 FastAPI Chatbot with Streaming")
st.write("Ask me anything!")

# Custom CSS for fade effect
st.markdown("""
<style>
.chat-chunk {
    display: inline-block;
    opacity: 0;
    animation: fadeIn 0.2s ease-in forwards;
}

@keyframes fadeIn {
    to {
        opacity: 1;
    }
}

.response-box {
    font-family: monospace;
    white-space: pre-wrap;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# Show chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        # Use markdown rendering for completed messages
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Type your message...")
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    payload = {
        "message": user_input,
        "session_id": st.session_state["session_id"]
    }

    start_time = time.time()

    try:
        with requests.post(API_URL, json=payload, stream=True) as response:
            response.raise_for_status()

            full_response = ""
            formatted_html = ""
            time_taken = time.time()
            with st.chat_message("assistant"):
                placeholder = st.empty()

                for chunk in response.iter_content(chunk_size=None):
                    decoded = chunk.decode("utf-8")
                    full_response += decoded

                    # Render streaming HTML
                    placeholder.markdown(full_response)

                # Final render as Markdown (better formatting)
                final_text = full_response + f"\n\n⏳ *Latency: { time_taken - start_time:.2f}s*"
                placeholder.markdown(final_text)

                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": final_text
                })

    except requests.exceptions.RequestException as e:
        error_message = f"⚠️ Error: {str(e)}"
        st.session_state["messages"].append({"role": "assistant", "content": error_message})
        with st.chat_message("assistant"):
            st.markdown(error_message)
