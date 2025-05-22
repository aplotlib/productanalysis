"""
AI Chat Module for Amazon Medical Device Listing Optimizer

Standalone AI chat interface for listing optimization advice, competitor analysis,
market research, and general Amazon selling guidance without requiring data uploads.

Author: Assistant  
Version: 2.0
"""

import streamlit as st
import logging
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chat configurations
CHAT_CONFIG = {
    'max_messages': 50,
    'context_window': 10,  # Number of previous messages to include for context
    'max_tokens_per_response': 1000,
    'temperature': 0.7
}

# Predefined expert personas
AI_PERSONAS = {
    'listing_optimizer': {
        'name': 'Amazon Listing Expert',
        'description': 'Specialist in Amazon SEO, keyword optimization, and conversion rate improvement',
        'system_prompt': """You are an expert Amazon listing optimization specialist with 10+ years of experience helping medical device companies maximize their e-commerce performance.

Your expertise includes:
- Amazon SEO and keyword research
- Product title and bullet point optimization  
- A+ Content and Enhanced Brand Content strategy
- Image optimization and visual storytelling
- Conversion rate optimization
- Competitor analysis and positioning
- Medical device compliance and regulations
- Customer review analysis and reputation management

Provide specific, actionable advice with concrete examples. Focus on medical device products when giving examples."""
    },
    
    'market_researcher': {
        'name': 'Market Research Analyst', 
        'description': 'Expert in Amazon market analysis, competitor research, and trend identification',
        'system_prompt': """You are a market research analyst specializing in Amazon marketplace dynamics and medical device industry trends.

Your expertise includes:
- Amazon marketplace analysis and trends
- Competitor research and positioning
- Market opportunity identification
- Pricing strategy and positioning
- Category analysis and market sizing
- Consumer behavior and purchase patterns
- Medical device market regulations and compliance
- Seasonal trends and demand forecasting

Provide data-driven insights with market context and strategic recommendations."""
    },
    
    'return_specialist': {
        'name': 'Return Reduction Expert',
        'description': 'Specialist in reducing Amazon return rates and improving customer satisfaction',
        'system_prompt': """You are a return reduction specialist who helps Amazon sellers minimize return rates and improve customer satisfaction for medical device products.

Your expertise includes:
- Return reason analysis and pattern identification
- Customer expectation management
- Product description optimization to reduce returns
- Image and video strategies to set proper expectations
- Packaging and unboxing experience optimization
- Quality control and defect prevention
- Customer service and communication strategies
- Medical device user education and instruction clarity

Focus on practical solutions that reduce returns while maintaining customer satisfaction."""
    },
    
    'general_consultant': {
        'name': 'Amazon Business Consultant',
        'description': 'General Amazon business strategy and growth advisor',
        'system_prompt': """You are a comprehensive Amazon business consultant specializing in medical device e-commerce growth and optimization.

Your expertise covers:
- Overall Amazon business strategy
- Brand building and positioning
- Inventory management and forecasting
- Advertising and PPC optimization
- International expansion strategies
- Compliance and regulatory considerations
- Business analytics and performance tracking
- Team building and process optimization

Provide strategic guidance with both short-term tactical advice and long-term growth strategies."""
    }
}

# Quick start prompts for users
QUICK_START_PROMPTS = {
    'listing_optimization': [
        "How can I improve my product title for better search visibility?",
        "What are the best practices for medical device bullet points?",
        "How do I optimize my main product image for higher conversion?",
        "What keywords should I target for [product type]?",
        "How can I improve my A+ Content for medical devices?"
    ],
    
    'market_research': [
        "What's the market opportunity for [product category] on Amazon?",
        "How do I analyze my competitors' listings effectively?",
        "What pricing strategy should I use for a new medical device?",
        "What are the current trends in [medical device category]?",
        "How do I identify underserved market segments?"
    ],
    
    'return_reduction': [
        "Why might customers be returning my [product type]?",
        "How can I reduce size-related returns for medical devices?",
        "What images should I include to set proper expectations?",
        "How do I write descriptions that prevent misunderstandings?",
        "What's the best way to explain product limitations clearly?"
    ],
    
    'general_advice': [
        "How do I launch a new medical device on Amazon successfully?",
        "What's the best advertising strategy for medical device products?",
        "How do I handle negative reviews professionally?",
        "What compliance considerations do I need for medical devices?",
        "How do I scale my Amazon medical device business?"
    ]
}

class AIAPIClient:
    """Enhanced API client for chat functionality"""
    
    def __init__(self):
        self.api_key = self._get_api_key()
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from Streamlit secrets or environment"""
        try:
            if hasattr(st, 'secrets') and 'openai_api_key' in st.secrets:
                return st.secrets['openai_api_key']
        except:
            pass
        
        import os
        return os.environ.get('OPENAI_API_KEY')
    
    def is_available(self) -> bool:
        """Check if API is available"""
        return self.api_key is not None
    
    def send_chat_message(self, messages: List[Dict[str, str]], 
                         temperature: float = 0.7, 
                         max_tokens: int = 1000) -> Dict[str, Any]:
        """Send chat message to OpenAI API"""
        
        if not self.api_key:
            return {
                'success': False,
                'error': 'API key not configured',
                'response': 'AI chat is not available. Please configure your OpenAI API key.'
            }
        
        try:
            payload = {
                "model": "gpt-4o",
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'response': result['choices'][0]['message']['content'],
                    'usage': result.get('usage', {})
                }
            else:
                return {
                    'success': False,
                    'error': f"API Error {response.status_code}",
                    'response': f"Sorry, I'm having trouble connecting right now. Error: {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"Chat API error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'response': "Sorry, I'm experiencing technical difficulties. Please try again."
            }

class ChatSession:
    """Manages individual chat session state and context"""
    
    def __init__(self, persona_key: str = 'general_consultant'):
        self.persona_key = persona_key
        self.messages = []
        self.session_start = datetime.now()
        self.total_tokens_used = 0
        self.api_client = AIAPIClient()
    
    def get_persona_info(self) -> Dict[str, str]:
        """Get current persona information"""
        return AI_PERSONAS.get(self.persona_key, AI_PERSONAS['general_consultant'])
    
    def add_user_message(self, message: str):
        """Add user message to conversation"""
        self.messages.append({
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Limit message history
        if len(self.messages) > CHAT_CONFIG['max_messages']:
            self.messages = self.messages[-CHAT_CONFIG['max_messages']:]
    
    def add_assistant_message(self, message: str, usage_info: Dict[str, Any] = None):
        """Add assistant message to conversation"""
        self.messages.append({
            'role': 'assistant', 
            'content': message,
            'timestamp': datetime.now().isoformat()
        })
        
        if usage_info:
            self.total_tokens_used += usage_info.get('total_tokens', 0)
    
    def get_conversation_context(self) -> List[Dict[str, str]]:
        """Get conversation context for API call"""
        persona = self.get_persona_info()
        
        # Start with system message
        context = [{'role': 'system', 'content': persona['system_prompt']}]
        
        # Add recent conversation history
        recent_messages = self.messages[-CHAT_CONFIG['context_window']:]
        for msg in recent_messages:
            context.append({
                'role': msg['role'],
                'content': msg['content']
            })
        
        return context
    
    def send_message(self, user_message: str) -> str:
        """Send message and get AI response"""
        
        # Add user message
        self.add_user_message(user_message)
        
        # Get conversation context
        context = self.get_conversation_context()
        
        # Call API
        result = self.api_client.send_chat_message(
            context,
            temperature=CHAT_CONFIG['temperature'],
            max_tokens=CHAT_CONFIG['max_tokens_per_response']
        )
        
        if result['success']:
            response = result['response']
            self.add_assistant_message(response, result.get('usage'))
            return response
        else:
            error_message = result['response']
            self.add_assistant_message(error_message)
            return error_message
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.messages = []
        self.total_tokens_used = 0
    
    def export_conversation(self) -> str:
        """Export conversation as text"""
        persona = self.get_persona_info()
        
        export_text = f"# Amazon Listing Optimization Chat Session\n"
        export_text += f"**Expert:** {persona['name']}\n"
        export_text += f"**Session Start:** {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n"
        export_text += f"**Total Messages:** {len(self.messages)}\n\n"
        
        for msg in self.messages:
            role = "You" if msg['role'] == 'user' else persona['name']
            timestamp = datetime.fromisoformat(msg['timestamp']).strftime('%H:%M:%S')
            export_text += f"**{role}** ({timestamp}):\n{msg['content']}\n\n"
        
        return export_text

class AIChatInterface:
    """Main chat interface component"""
    
    def __init__(self):
        self.api_client = AIAPIClient()
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize chat session state"""
        if 'chat_session' not in st.session_state:
            st.session_state.chat_session = ChatSession()
        
        if 'show_quick_prompts' not in st.session_state:
            st.session_state.show_quick_prompts = True
    
    def render_chat_interface(self):
        """Render the main chat interface"""
        
        st.markdown("## 🤖 AI Listing Optimization Chat")
        st.markdown("Get expert advice on Amazon listing optimization, market research, and strategy - no data upload required!")
        
        # Check API availability
        if not self.api_client.is_available():
            st.error("🔑 AI Chat requires OpenAI API key configuration")
            st.info("Add your OpenAI API key in Streamlit secrets to enable AI chat functionality")
            return
        
        # Persona selector
        self._render_persona_selector()
        
        # Quick start prompts
        if st.session_state.show_quick_prompts:
            self._render_quick_prompts()
        
        # Chat history
        self._render_chat_history()
        
        # Chat input
        self._render_chat_input()
        
        # Chat controls
        self._render_chat_controls()
    
    def _render_persona_selector(self):
        """Render AI persona selection"""
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            current_persona = st.session_state.chat_session.persona_key
            persona_options = list(AI_PERSONAS.keys())
            persona_labels = [AI_PERSONAS[key]['name'] for key in persona_options]
            
            selected_index = persona_options.index(current_persona) if current_persona in persona_options else 0
            
            new_persona = st.selectbox(
                "Choose your AI expert:",
                options=persona_options,
                format_func=lambda x: AI_PERSONAS[x]['name'],
                index=selected_index,
                help="Select the type of expert advice you need"
            )
            
            # Update persona if changed
            if new_persona != current_persona:
                st.session_state.chat_session.persona_key = new_persona
                st.rerun()
        
        with col2:
            persona_info = st.session_state.chat_session.get_persona_info()
            st.markdown(f"**Expert:** {persona_info['name']}")
            st.caption(persona_info['description'])
    
    def _render_quick_prompts(self):
        """Render quick start prompts"""
        
        with st.expander("💡 Quick Start Prompts", expanded=True):
            persona_key = st.session_state.chat_session.persona_key
            
            # Get prompts for current persona or general prompts
            if persona_key == 'listing_optimizer':
                prompts = QUICK_START_PROMPTS['listing_optimization']
            elif persona_key == 'market_researcher':
                prompts = QUICK_START_PROMPTS['market_research']
            elif persona_key == 'return_specialist':
                prompts = QUICK_START_PROMPTS['return_reduction']
            else:
                prompts = QUICK_START_PROMPTS['general_advice']
            
            # Display prompts as clickable buttons
            cols = st.columns(2)
            for i, prompt in enumerate(prompts):
                col = cols[i % 2]
                with col:
                    if st.button(prompt, key=f"prompt_{i}", use_container_width=True):
                        # Send the selected prompt
                        st.session_state.selected_prompt = prompt
                        st.session_state.show_quick_prompts = False
                        st.rerun()
    
    def _render_chat_history(self):
        """Render chat conversation history"""
        
        chat_container = st.container()
        
        with chat_container:
            if not st.session_state.chat_session.messages:
                persona_info = st.session_state.chat_session.get_persona_info()
                st.info(f"👋 Hi! I'm your {persona_info['name']}. How can I help you optimize your Amazon listings today?")
            else:
                # Display conversation
                for message in st.session_state.chat_session.messages:
                    role = message['role']
                    content = message['content']
                    
                    if role == 'user':
                        with st.chat_message("user"):
                            st.markdown(content)
                    else:
                        with st.chat_message("assistant"):
                            st.markdown(content)
    
    def _render_chat_input(self):
        """Render chat input interface"""
        
        # Handle selected quick prompt
        if 'selected_prompt' in st.session_state:
            prompt = st.session_state.selected_prompt
            del st.session_state.selected_prompt
            self._send_message(prompt)
            return
        
        # Chat input
        user_input = st.chat_input("Ask me anything about Amazon listing optimization...")
        
        if user_input:
            self._send_message(user_input)
    
    def _send_message(self, message: str):
        """Send message and get AI response"""
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(message)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_session.send_message(message)
            st.markdown(response)
        
        # Rerun to update chat history
        st.rerun()
    
    def _render_chat_controls(self):
        """Render chat control buttons"""
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🗑️ Clear Chat", help="Clear conversation history"):
                st.session_state.chat_session.clear_conversation()
                st.session_state.show_quick_prompts = True
                st.rerun()
        
        with col2:
            if st.button("💡 Show Prompts", help="Show quick start prompts"):
                st.session_state.show_quick_prompts = True
                st.rerun()
        
        with col3:
            # Token usage info
            tokens_used = st.session_state.chat_session.total_tokens_used
            if tokens_used > 0:
                st.caption(f"Tokens used: {tokens_used:,}")
        
        with col4:
            # Export conversation
            if len(st.session_state.chat_session.messages) > 0:
                conversation_text = st.session_state.chat_session.export_conversation()
                st.download_button(
                    label="📥 Export Chat",
                    data=conversation_text,
                    file_name=f"ai_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown",
                    help="Download conversation as text file"
                )

# Standalone chat page function
def render_ai_chat_page():
    """Render standalone AI chat page"""
    
    st.set_page_config(
        page_title="AI Chat - Amazon Listing Optimizer",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize and render chat interface
    chat_interface = AIChatInterface()
    chat_interface.render_chat_interface()

# Export main classes
__all__ = ['AIChatInterface', 'ChatSession', 'AI_PERSONAS', 'render_ai_chat_page']
