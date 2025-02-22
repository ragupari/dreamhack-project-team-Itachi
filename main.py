import streamlit as st
import asyncio
from chat import agent

async def get_response(query, messages):
    response = await agent.run(query, message_history=messages)
    return response.data, response.all_messages()

async def local_test():
    st.title("Chat with the Agent")

    # Initialize message history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    user_input = st.text_input("You:", "")

    if user_input:
        async def run_agent():
            response, messages = await get_response(user_input, st.session_state.messages)
            st.session_state.messages = messages  # Update message history
            st.session_state.last_response = response  # Store last response

        await run_agent()  # Ensure this is awaited if it's an async function

    if 'last_response' in st.session_state:
        st.write("Agent: " + st.session_state.last_response)

    if st.button("Quit"):
        st.session_state.messages = []
        st.session_state.last_response = ""
        st.write("Session Ended. Thank you for chatting!")

if __name__ == "__main__":
    asyncio.run(local_test())