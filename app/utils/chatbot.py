"""
Financial Chatbot using Groq, LangChain, and Google Gemma Model
Provides intelligent responses to financial and loan-related queries
"""
import os
from typing import Optional, List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class FinancialChatbot:
    """
    Financial chatbot powered by Groq's Gemma model
    Specialized in answering loan and financial queries
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the financial chatbot
        
        Args:
            api_key: Groq API key (if None, will try to get from environment)
        """
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "Groq API key not found. Please set GROQ_API_KEY environment variable "
                "or pass it to the constructor."
            )
        
        # Initialize Groq chat model with Gemma
        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name="gemma-7b-it",  # Google Gemma 7B Instruct model
            temperature=0.7,
            max_tokens=1024,
        )
        
        # Create financial-focused system prompt
        self.system_prompt = """You are a professional financial advisor and loan specialist assistant. 
Your role is to help users understand loan eligibility, financial requirements, and provide guidance on loan applications.

Key Responsibilities:
1. Answer questions about loan eligibility criteria
2. Explain financial concepts in simple, clear terms
3. Provide guidance on improving loan eligibility
4. Explain credit scores, debt-to-income ratios, and other financial metrics
5. Help users understand loan terms and requirements

Guidelines:
- Always be professional, helpful, and empathetic
- Provide accurate, clear explanations
- If you don't know something, admit it rather than guessing
- Focus on educational and advisory information
- Never provide specific financial advice that could be considered personalized investment advice
- Encourage users to consult with qualified financial professionals for complex situations

When discussing loan eligibility:
- Explain factors that affect loan approval (credit score, income, debt-to-income ratio, employment history)
- Provide general guidance on improving eligibility
- Explain how different loan types work
- Discuss the importance of credit history

Remember: You are an AI assistant providing general information and guidance, not a substitute for professional financial advice."""

        # Initialize conversation history
        self.conversation_history: List = []
    
    def chat(self, message: str) -> str:
        """
        Get response from the chatbot
        
        Args:
            message: User's message/query
            
        Returns:
            Bot's response
        """
        try:
            if not message or not message.strip():
                return "Please provide a question or message. I'm here to help with loan and financial queries!"
            
            # Build messages list with system prompt and conversation history
            messages = [SystemMessage(content=self.system_prompt)]
            
            # Add conversation history
            messages.extend(self.conversation_history)
            
            # Add current user message
            messages.append(HumanMessage(content=message.strip()))
            
            # Get response from the model
            response = self.llm.invoke(messages)
            
            # Extract response text
            if hasattr(response, 'content'):
                response_text = response.content
            elif isinstance(response, str):
                response_text = response
            else:
                response_text = str(response)
            
            # Update conversation history (keep last 10 exchanges to manage context)
            self.conversation_history.append(HumanMessage(content=message.strip()))
            self.conversation_history.append(AIMessage(content=response_text))
            
            # Limit conversation history to last 10 messages (5 exchanges)
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return response_text.strip()
            
        except Exception as e:
            # Return a helpful error message
            return f"I apologize, but I encountered an error processing your request. Please try again or rephrase your question. Error: {str(e)}"
    
    def reset_conversation(self):
        """
        Reset the conversation memory
        """
        self.conversation_history = []
    
    def get_conversation_summary(self) -> dict:
        """
        Get summary of the current conversation
        
        Returns:
            Dictionary with conversation summary
        """
        return {
            'conversation_history': len(self.conversation_history),
            'has_memory': len(self.conversation_history) > 0
        }

# Global chatbot instance
_chatbot_instance = None

def get_chatbot(api_key: Optional[str] = None) -> FinancialChatbot:
    """
    Get or create the global chatbot instance
    
    Args:
        api_key: Groq API key (optional)
        
    Returns:
        FinancialChatbot instance
    """
    global _chatbot_instance
    
    if _chatbot_instance is None:
        try:
            _chatbot_instance = FinancialChatbot(api_key=api_key)
        except Exception as e:
            print(f"Warning: Could not initialize chatbot: {e}")
            print("Chatbot will return placeholder responses.")
            return None
    
    return _chatbot_instance

def reset_chatbot():
    """
    Reset the global chatbot instance
    """
    global _chatbot_instance
    if _chatbot_instance:
        _chatbot_instance.reset_conversation()

