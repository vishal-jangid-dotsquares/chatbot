from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
import initial


class CustomChatMemory:
    
    def __init__(self, user_id, expiry=6000):
        """Initialize Redis client and expiry settings."""
        self.user_summary_key = f"{user_id}:last_summary"
        self.user_last_message_key = f"{user_id}:last_message"
        self.user_last_filter_tag_key = f"{user_id}:last_filter_tag"
        self.user_last_division = f"{user_id}:last_division"

        self.redis_client = initial.REDIS_CLIENT
        self.expiry = expiry  # Expiry in seconds (default: 5 minutes)

    async def get_conversation(self):
        """Retrieve the summarized conversation"""
        summary_coroutine = self.redis_client.get(self.user_summary_key)
        summary = await summary_coroutine if summary_coroutine else ""
        return summary
  
    async def get_last_message(self):
        """Retrieve the last saved message."""
        message_coroutine = self.redis_client.get(self.user_last_message_key)
        last_message = await message_coroutine if message_coroutine else ""
        return last_message
  
    async def get_last_filter_tag(self):
        """Retrieve the last saved filter tag."""
        filter_tag_coroutine = self.redis_client.get(self.user_last_filter_tag_key)
        last_filter_tag = await filter_tag_coroutine if filter_tag_coroutine else ""
        return last_filter_tag
  
    async def get_last_division(self):
        """Retrieve the last saved division."""
        division_coroutine = self.redis_client.get(self.user_last_division)
        last_division = await division_coroutine if division_coroutine else ""
        return last_division

    async def add_filter_tag(self, filter_tag:str):
        await self.redis_client.set(self.user_last_filter_tag_key, filter_tag, ex=self.expiry)

    async def add_division(self, division:str):
        await self.redis_client.set(self.user_last_division, division, ex=self.expiry)

    async def add_memory(self, user_message: str, bot_response: str):
        """Store the last message and update the summary."""

        # Save last message (replace old one)
        await self.redis_client.set(self.user_last_message_key, user_message, ex=self.expiry)

        # Retrieve existing summary
        old_summary = await self.get_conversation()
        
        # Generate a new summary combining old summary + latest conversation
        conversation = f"User: {user_message} | Bot: {bot_response}"
        new_summary = await self.summarize_text(old_summary, conversation)
        
        # Save updated summary (replace old one)
        await self.redis_client.set(self.user_summary_key, new_summary, ex=self.expiry)

    async def summarize_text(self, old_summary: str, new_chat: str):
        """Summarize given conversations using LangChain with a strict 200-word limit."""
        chain = load_summarize_chain(
            initial.SUMMERIZING_MODEL, 
            chain_type="stuff",
        )
        
        # Combine old summary with new chat history
        combined_text = f"{old_summary} {new_chat}".strip()
        prompt = initial.PRE_PROMPTS['memory'].format(
            input_text = combined_text
        )
        
        docs = [Document(page_content=prompt)]
        response = await chain.ainvoke({"input_documents": docs})
        summary = response["output_text"]
        
        return summary  

