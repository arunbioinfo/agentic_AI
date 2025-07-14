import streamlit as st
import os
import tempfile
import json
from datetime import datetime
from pathlib import Path
import requests
import PyPDF2
import docx
from typing import List, Dict, Any
import hashlib

# Configure Streamlit page
st.set_page_config(
    page_title="🤖 Agentic AI Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }
    
    .main-header h1 {
        background: linear-gradient(135deg, #40e0d0, #48cae4, #0077be);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .upload-area {
        border: 2px dashed #40e0d0;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: rgba(64, 224, 208, 0.05);
        margin: 1rem 0;
    }
    
    .search-results {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    
    .result-item {
        background: rgba(64, 224, 208, 0.1);
        border: 1px solid rgba(64, 224, 208, 0.3);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .stats-card {
        background: linear-gradient(135deg, rgba(64, 224, 208, 0.1), rgba(72, 202, 228, 0.1));
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(64, 224, 208, 0.3);
    }
    
    .file-item {
        background: rgba(64, 224, 208, 0.1);
        border: 1px solid rgba(64, 224, 208, 0.3);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "llama3.2:3b"  # Free 3B parameter model
        self.available_models = []
    
    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                self.available_models = [model['name'] for model in models_data.get('models', [])]
                return True
            return False
        except Exception as e:
            st.error(f"Failed to connect to Ollama: {str(e)}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return self.available_models
    
    def model_exists(self, model_name: str) -> bool:
        """Check if a specific model exists"""
        return model_name in self.available_models
    
    def pull_model(self, model_name: str = None) -> bool:
        """Pull the model if not available"""
        if model_name is None:
            model_name = self.model
            
        try:
            st.info(f"Pulling model {model_name}... This may take a few minutes.")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=600,
                stream=True
            )
            
            if response.status_code == 200:
                # Parse streaming response
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if 'status' in data:
                                st.info(f"Status: {data['status']}")
                            if data.get('completed', False):
                                st.success(f"Model {model_name} pulled successfully!")
                                return True
                        except:
                            continue
                return True
            else:
                st.error(f"Failed to pull model: {response.status_code}")
                return False
        except Exception as e:
            st.error(f"Error pulling model: {str(e)}")
            return False
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using Ollama"""
        try:
            # Check if service is available
            if not self.is_available():
                return "❌ Ollama service is not running. Please start Ollama first."
            
            # Check if model exists
            if not self.model_exists(self.model):
                available_models = self.get_available_models()
                if available_models:
                    # Use the first available model
                    self.model = available_models[0]
                    st.warning(f"Using available model: {self.model}")
                else:
                    return f"❌ No models available. Please run: `ollama pull {self.model}`"
            
            full_prompt = f"""Context: {context}

User Query: {prompt}

Based on the provided context, please provide a comprehensive and accurate response. If the context doesn't contain relevant information, please state that clearly.

Response:"""

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 1000
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "No response generated")
            elif response.status_code == 404:
                return f"❌ Model '{self.model}' not found. Available models: {', '.join(self.available_models) if self.available_models else 'None'}"
            else:
                return f"❌ Error {response.status_code}: {response.text}"
                
        except requests.exceptions.ConnectionError:
            return "❌ Cannot connect to Ollama. Please ensure Ollama is running with: `ollama serve`"
        except requests.exceptions.Timeout:
            return "❌ Request timed out. The model might be too slow or overloaded."
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"

class DocumentProcessor:
    """Process different document types"""
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    
    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            return f"Error reading DOCX: {str(e)}"
    
    @staticmethod
    def extract_text_from_txt(file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            return f"Error reading TXT: {str(e)}"
    
    @classmethod
    def process_document(cls, file_path: str, file_type: str) -> str:
        """Process document based on file type"""
        if file_type == "application/pdf":
            return cls.extract_text_from_pdf(file_path)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return cls.extract_text_from_docx(file_path)
        elif file_type == "text/plain":
            return cls.extract_text_from_txt(file_path)
        else:
            return "Unsupported file type"

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'search_count' not in st.session_state:
        st.session_state.search_count = 0
    if 'ollama_client' not in st.session_state:
        st.session_state.ollama_client = OllamaClient()

def display_header():
    """Display the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Agentic AI Dashboard</h1>
        <p style="font-size: 1.2rem; opacity: 0.8;">Module 1: Intelligent Document Upload & Search with Free LLM</p>
    </div>
    """, unsafe_allow_html=True)

def display_ollama_status():
    """Display Ollama connection status"""
    with st.sidebar:
        st.markdown("### 🔧 LLM Status")
        
        ollama_client = st.session_state.ollama_client
        
        if ollama_client.is_available():
            st.success("✅ Ollama is running")
            
            available_models = ollama_client.get_available_models()
            if available_models:
                #st.info(f"📦 Available models: {', '.join(available_models)}")
                
                # Model selector
                selected_model = st.selectbox(
                    "Select Model:",
                    available_models,
                    index=0 if ollama_client.model in available_models else 0
                )
                
                if selected_model != ollama_client.model:
                    ollama_client.model = selected_model
                    #st.success(f"Model changed to: {selected_model}")
            else:
                st.warning("⚠️ No models found")
                
                if st.button("📥 Pull Llama 3.2"):
                    with st.spinner("Downloading model..."):
                        if ollama_client.pull_model("llama3.2:3b"):
                            st.success("✅ Model downloaded successfully!")
                            st.experimental_rerun()
                        else:
                            st.error("❌ Failed to download model")
        else:
            st.error("❌ Ollama not available")
            st.markdown("""
            **Quick Setup:**
            
            1. **Install Ollama:**
            ```bash
            # Linux/Mac
            curl -fsSL https://ollama.ai/install.sh | sh
            
            # Windows
            # Download from https://ollama.ai
            ```
            
            2. **Start Ollama:**
            ```bash
            ollama serve
            ```
            
            3. **Download a model:**
            ```bash
            ollama pull llama3.2:3b
            # or try smaller models:
            ollama pull llama3.2:1b
            ollama pull qwen2.5:3b
            ```
            """)
            
            if st.button("🔄 Retry Connection"):
                st.experimental_rerun()
        
        # Additional troubleshooting
        # with st.expander("🔍 Troubleshooting"):
        #     st.markdown("""
        #     **Common Issues:**
            
        #     1. **Error 404**: Model not found
        #        - Run: `ollama pull llama3.2:3b`
        #        - Or try: `ollama pull llama3.2:1b` (smaller)
            
        #     2. **Connection Error**: Ollama not running
        #        - Run: `ollama serve` in terminal
        #        - Check if port 11434 is free
            
        #     3. **Timeout**: Model too slow
        #        - Try smaller model (1b instead of 3b)
        #        - Increase timeout in settings
            
        #     **Alternative Models:**
        #     - `llama3.2:1b` (faster, smaller)
        #     - `qwen2.5:3b` (good alternative)
        #     - `phi3:mini` (very fast)
        #     """)
        
        # # System info
        # st.markdown("### 💻 System Info")
        # st.info(f"Ollama URL: {ollama_client.base_url}")
        # st.info(f"Current Model: {ollama_client.model}")
        
        # # Test connection button
        # if st.button("🧪 Test Connection"):
        #     if ollama_client.is_available():
        #         test_response = ollama_client.generate_response("Hello, are you working?", "")
        #         st.write("**Test Response:**")
        #         st.write(test_response)

def handle_file_upload():
    """Handle file upload functionality"""
    st.markdown("### 📤 Article Upload")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="Upload PDF, DOCX, or TXT files"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Check if file already exists
            file_hash = hashlib.md5(uploaded_file.read()).hexdigest()
            uploaded_file.seek(0)  # Reset file pointer
            
            existing_file = next((f for f in st.session_state.uploaded_files if f['hash'] == file_hash), None)
            
            if not existing_file:
                # Save file temporarily and process
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                
                # Extract text
                text_content = DocumentProcessor.process_document(tmp_file_path, uploaded_file.type)
                
                # Store file info
                file_info = {
                    'name': uploaded_file.name,
                    'size': uploaded_file.size,
                    'type': uploaded_file.type,
                    'upload_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'content': text_content,
                    'hash': file_hash,
                    'path': tmp_file_path
                }
                
                st.session_state.uploaded_files.append(file_info)
                st.success(f"✅ {uploaded_file.name} uploaded successfully!")
                
                # Clean up
                os.unlink(tmp_file_path)
            else:
                st.warning(f"⚠️ {uploaded_file.name} already exists!")

def display_uploaded_files():
    """Display list of uploaded files"""
    if st.session_state.uploaded_files:
        st.markdown("### 📚 Uploaded Files")
        
        for i, file_info in enumerate(st.session_state.uploaded_files):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"""
                <div class="file-item">
                    <strong>{file_info['name']}</strong><br>
                    <small>Size: {file_info['size']:,} bytes | Uploaded: {file_info['upload_date']}</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.button("🔍 Preview", key=f"preview_{i}"):
                    with st.expander(f"Preview: {file_info['name']}", expanded=True):
                        st.text_area("Content", file_info['content'][:1000] + "..." if len(file_info['content']) > 1000 else file_info['content'], height=200)
            
            with col3:
                if st.button("🗑️ Delete", key=f"delete_{i}"):
                    st.session_state.uploaded_files.pop(i)
                    st.experimental_rerun()

def handle_search():
    """Handle AI-powered search functionality"""
    st.markdown("### 🔍 AI-Powered Search")
    
    # Pre-check Ollama status
    ollama_client = st.session_state.ollama_client
    
    # Status indicator
    if ollama_client.is_available():
        available_models = ollama_client.get_available_models()
        if available_models:
            st.success(f"✅ Ready to search | Model: {ollama_client.model}")
        else:
            st.error("❌ No models available - check sidebar for setup")
    else:
        st.error("❌ Ollama not connected - check sidebar for setup")
    
    search_query = st.text_input(
        "Enter your search query or question:",
        placeholder="e.g., 'Summarize the main findings' or 'What are the key conclusions?'",
        help="Ask questions about your uploaded documents"
    )
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        max_chars = st.slider("Max context characters", 1000, 20000, 8000, 1000)
        temperature = st.slider("Response creativity", 0.1, 1.0, 0.7, 0.1)
        
        # Quick test button
        if st.button("🧪 Quick Test"):
            if ollama_client.is_available():
                test_response = ollama_client.generate_response("Say hello", "")
                st.write("**Test Result:**", test_response)
            else:
                st.error("Ollama not available for testing")
    
    if st.button("🚀 Search with AI", type="primary"):
        if not search_query.strip():
            st.error("Please enter a search query!")
            return
        
        if not st.session_state.uploaded_files:
            st.error("Please upload documents first!")
            return
        
        # Detailed status check
        if not ollama_client.is_available():
            st.error("❌ Ollama is not running. Please:")
            st.code("ollama serve")
            return
        
        available_models = ollama_client.get_available_models()
        if not available_models:
            st.error("❌ No models found. Please install a model:")
            st.code("ollama pull llama3.2:3b")
            if st.button("📥 Auto-install model"):
                with st.spinner("Installing model..."):
                    if ollama_client.pull_model():
                        st.success("Model installed! Please retry search.")
                        st.experimental_rerun()
            return
        
        if not ollama_client.model_exists(ollama_client.model):
            st.warning(f"Model {ollama_client.model} not found. Available: {', '.join(available_models)}")
            if available_models:
                ollama_client.model = available_models[0]
                st.info(f"Switched to: {ollama_client.model}")
        
        # Increment search count
        st.session_state.search_count += 1
        
        # Combine all document content
        combined_content = "\n\n".join([
            f"Document: {file_info['name']}\n{file_info['content']}"
            for file_info in st.session_state.uploaded_files
        ])
        
        # Limit content size
        if len(combined_content) > max_chars:
            combined_content = combined_content[:max_chars] + "\n\n[Content truncated due to length...]"
        
        with st.spinner("🤖 AI is analyzing your documents..."):
            # Generate response using Ollama
            response = ollama_client.generate_response(search_query, combined_content)
        
        # Display results
        st.markdown("### 📊 AI Analysis Results")
        
        # Check if response indicates an error
        if response.startswith("❌"):
            st.error(response)
            
            # Provide specific help based on error type
            if "404" in response or "not found" in response.lower():
                st.markdown("""
                **Solution Steps:**
                1. Check available models in sidebar
                2. Install a model: `ollama pull llama3.2:3b`
                3. Restart the search
                """)
            elif "connection" in response.lower():
                st.markdown("""
                **Solution Steps:**
                1. Start Ollama: `ollama serve`
                2. Wait for it to start (check http://localhost:11434)
                3. Refresh this page
                """)
        else:
            st.markdown(f"""
            <div class="search-results">
                <h4>🎯 Query: {search_query}</h4>
                <div class="result-item">
                    <h5>🤖 AI Response:</h5>
                    <p>{response}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional insights
        with st.expander("📈 Additional Insights"):
            st.write("**Documents Analyzed:**")
            for file_info in st.session_state.uploaded_files:
                st.write(f"• {file_info['name']} ({file_info['size']:,} bytes)")
            
            st.write(f"**Total Content Length:** {len(combined_content):,} characters")
            st.write(f"**Model Used:** {ollama_client.model}")
            st.write(f"**Available Models:** {', '.join(available_models) if available_models else 'None'}")

def display_statistics():
    """Display dashboard statistics"""
    st.markdown("### 📊 Dashboard Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stats-card">
            <div style="font-size: 2rem;">📚</div>
            <div style="font-size: 1.5rem; font-weight: bold;">{len(st.session_state.uploaded_files)}</div>
            <div>Total Documents</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stats-card">
            <div style="font-size: 2rem;">🔍</div>
            <div style="font-size: 1.5rem; font-weight: bold;">{st.session_state.search_count}</div>
            <div>AI Searches</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_size = sum(file_info['size'] for file_info in st.session_state.uploaded_files)
        st.markdown(f"""
        <div class="stats-card">
            <div style="font-size: 2rem;">💾</div>
            <div style="font-size: 1.5rem; font-weight: bold;">{total_size:,}</div>
            <div>Total Bytes</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        ollama_status = "Online" if st.session_state.ollama_client.is_available() else "Offline"
        status_color = "#2ecc71" if ollama_status == "Online" else "#e74c3c"
        st.markdown(f"""
        <div class="stats-card">
            <div style="font-size: 2rem;">🤖</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: {status_color};">{ollama_status}</div>
            <div>LLM Status</div>
        </div>
        """, unsafe_allow_html=True)

def display_instructions():
    """Display setup and usage instructions"""
    with st.sidebar:
        st.markdown("### 📋 Instructions")
        
        # with st.expander("🚀 Setup Guide"):
        #     st.markdown("""
        #     **1. Install Ollama:**
        #     ```bash
        #     # Linux/Mac
        #     curl -fsSL https://ollama.ai/install.sh | sh
            
        #     # Windows
        #     # Download from https://ollama.ai/download
        #     ```
            
        #     **2. Start Ollama:**
        #     ```bash
        #     ollama serve
        #     ```
            
        #     **3. Pull the model:**
        #     ```bash
        #     ollama pull llama3.2:3b
        #     ```
            
        #     **4. Install Python dependencies:**
        #     ```bash
        #     pip install streamlit PyPDF2 python-docx requests
        #     ```
        #     """)
        
        with st.expander("📖 Usage Guide"):
            st.markdown("""
            **1. Upload Documents:**
            - Support for PDF, DOCX, TXT files
            - Multiple files can be uploaded
            - Files are processed automatically
            
            **2. Search & Query:**
            - Ask questions about your documents
            - Request summaries or analysis
            - Get AI-powered insights
            
            **3. Examples:**
            - "Summarize the main findings"
            - "What are the key conclusions?"
            - "Compare the different approaches"
            - "Extract important statistics"
            """)

def main():
    """Main application function"""
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Display Ollama status in sidebar
    display_ollama_status()
    display_instructions()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        handle_file_upload()
        display_uploaded_files()
    
    with col2:
        handle_search()
    
    # Statistics section
    display_statistics()
    
    # Footer
    st.markdown("---")
    st.markdown("**🤖 Agentic AI Dashboard** - Powered by Ollama & Llama 3.2 | Built with Streamlit")

if __name__ == "__main__":
    main()
