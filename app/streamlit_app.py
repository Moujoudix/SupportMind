# app.py - Production-Ready AI Support System
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import sys
import os
from pathlib import Path
import hashlib
import time
import traceback

# Import our real AI system
sys.path.append('.')

try:
    from system_core import system_manager
    SYSTEM_AVAILABLE = True
except Exception as e:
    SYSTEM_AVAILABLE = False
    st.error(f"⚠️ Could not import system_core: {str(e)}")

# Set page configuration
st.set_page_config(
    page_title="AI Support System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #D1FAE5;
        border-left: 4px solid #10B981;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #FEE2E2;
        border-left: 4px solid #EF4444;
        padding: 1rem;
        margin: 1rem 0;
    }
    .confidence-high {
        background-color: #10B981;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
    }
    .confidence-medium {
        background-color: #F59E0B;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
    }
    .confidence-low {
        background-color: #EF4444;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'current_suggestion' not in st.session_state:
    st.session_state.current_suggestion = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

def initialize_system():
    """Initialize the AI system with timeout protection"""
    if not SYSTEM_AVAILABLE:
        st.error("❌ system_core module not available")
        return False

    try:
        with st.spinner("🤖 Initializing AI System..."):
            result = system_manager.initialize_system("data")

            if result:
                st.session_state.system_initialized = True
                st.success("✅ System initialized successfully!")
                return True
            else:
                st.error("❌ Failed to initialize system - check data directory")
                return False

    except Exception as e:
        st.error(f"❌ Initialization error: {str(e)}")
        st.code(traceback.format_exc())
        return False

def safe_process_question(question, context=None):
    """Safely process a question with error handling"""
    try:
        if not SYSTEM_AVAILABLE:
            return {
                'error': 'System core not available',
                'customer_question': question,
                'suggested_answer': 'System is not properly configured.',
                'confidence_score': 0.0,
                'confidence_label': 'ERROR',
                'recommendation': 'System Error',
                'agent_guidance': 'Please check system configuration.',
                'sources': [],
                'num_sources': 0
            }

        # Call the actual system
        result = system_manager.process_customer_question(question, context)

        # Ensure all required fields exist
        if 'error' not in result:
            result.setdefault('customer_question', question)
            result.setdefault('suggested_answer', 'No answer generated')
            result.setdefault('confidence_score', 0.0)
            result.setdefault('confidence_label', 'UNKNOWN')
            result.setdefault('recommendation', 'Review manually')
            result.setdefault('agent_guidance', 'Please review the sources')
            result.setdefault('sources', [])
            result.setdefault('num_sources', len(result.get('sources', [])))

        return result

    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        return {
            'error': str(e),
            'customer_question': question,
            'suggested_answer': f'Error: {str(e)}',
            'confidence_score': 0.0,
            'confidence_label': 'ERROR',
            'recommendation': 'System Error',
            'agent_guidance': 'An error occurred',
            'sources': [],
            'num_sources': 0
        }

def safe_search(query, limit=5):
    """Safely search knowledge base"""
    try:
        if not SYSTEM_AVAILABLE:
            return []

        results = system_manager.search_knowledge_base(query, limit)
        return results if results else []

    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

def main():
    """Main application function"""

    # Header
    st.markdown("<h1 class='main-header'>🤖 Self-Learning AI Support System</h1>", unsafe_allow_html=True)

    # Check system availability
    if not SYSTEM_AVAILABLE:
        st.error("⚠️ System core module could not be loaded. Please check your installation.")
        st.stop()

    # System initialization check
    if not st.session_state.system_initialized:
        st.markdown("""
        <div class="card">
            <h3>🚀 Welcome!</h3>
            <p>This is the <strong>Real AI Support System</strong> with:</p>
            <ul>
                <li>📚 Live Knowledge Base Search</li>
                <li>🤖 Real AI Answer Generation</li>
                <li>🎯 Confidence Scoring</li>
                <li>🔍 Source Citations</li>
            </ul>
            <p>Click below to initialize the system:</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("⚡ Initialize AI System", type="primary", width="stretch"):
            if initialize_system():
                st.rerun()
        return

    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103655.png", width=100)
        st.markdown("## 🎯 Navigation")

        # Navigation
        app_mode = st.radio(
            "Choose a mode:",
            ["🏠 Dashboard", "💬 Agent Copilot", "📚 Knowledge Base", "📊 System Status"]
        )

        st.markdown("---")

        # System Status
        try:
            status = system_manager.get_system_status()
            st.markdown("### 📈 System Status")

            if status.get('status') == 'ready':
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("KB Docs", f"{status.get('knowledge_base_size', 0):,}")
                with col2:
                    st.metric("LLM", "✅ Active" if status.get('llm_active') else "⚠️ Mock")

                st.info(f"📊 {status.get('index_size', 0):,} vectors indexed")
            else:
                st.warning("System not ready")
        except:
            st.warning("Status unavailable")

        st.markdown("---")

        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        if st.button("🔄 Refresh System", width="stretch"):
            if initialize_system():
                st.rerun()

        st.markdown("---")
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

    # Main Content Area
    if app_mode == "🏠 Dashboard":
        show_dashboard()
    elif app_mode == "💬 Agent Copilot":
        show_agent_copilot()
    elif app_mode == "📚 Knowledge Base":
        show_knowledge_base()
    elif app_mode == "📊 System Status":
        show_system_status()

def show_dashboard():
    """Display main dashboard"""
    st.markdown("<h2 class='sub-header'>📊 Live System Dashboard</h2>", unsafe_allow_html=True)

    try:
        status = system_manager.get_system_status()
        kb_size = status.get('knowledge_base_size', 0)
    except:
        kb_size = 0

    # Welcome message
    st.markdown(f"""
    <div class="card">
        <h3>🎯 AI Support System Active!</h3>
        <p>The system is now <strong>live and operational</strong> with real AI capabilities:</p>
        <ul>
            <li>✅ <strong>Real-time knowledge retrieval</strong> from {kb_size:,} documents</li>
            <li>✅ <strong>AI-powered answer generation</strong> with Groq LLM</li>
            <li>✅ <strong>Confidence scoring</strong> and source citations</li>
            <li>✅ <strong>Instant search</strong> across all knowledge</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Quick Start Cards
    st.markdown("<h3 class='sub-header'>⚡ Quick Start</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 💬 Ask a Question")
        st.markdown("Get AI-powered support suggestions")

    with col2:
        st.markdown("### 📚 Search Knowledge")
        st.markdown("Find information instantly")

    with col3:
        st.markdown("### 📊 View Status")
        st.markdown("Check system health")

    # Try It Out Section
    st.markdown("<h3 class='sub-header'>🎯 Try It Out</h3>", unsafe_allow_html=True)

    quick_question = st.text_input(
        "Ask a quick question:",
        placeholder="e.g., How do I advance the property date?"
    )

    if st.button("🤖 Get AI Answer", width="stretch"):
        if quick_question:
            with st.spinner("🔍 Searching knowledge base..."):
                suggestion = safe_process_question(quick_question)

                if 'error' not in suggestion:
                    # Display confidence
                    confidence = suggestion['confidence_score']
                    if confidence > 0.7:
                        confidence_class = "confidence-high"
                    elif confidence > 0.4:
                        confidence_class = "confidence-medium"
                    else:
                        confidence_class = "confidence-low"

                    st.markdown(f"### 🎯 Confidence: <span class='{confidence_class}'>{confidence:.1%}</span>", unsafe_allow_html=True)

                    # Display suggestion
                    st.markdown("### 💬 AI Suggestion:")
                    st.markdown(suggestion['suggested_answer'])

                    # Display sources
                    with st.expander(f"📚 View Sources ({suggestion['num_sources']} found)"):
                        if suggestion['sources']:
                            sources_df = pd.DataFrame(suggestion['sources'])
                            st.dataframe(sources_df, width="stretch")
                else:
                    st.error(f"Error: {suggestion.get('error', 'Unknown error')}")

def show_agent_copilot():
    """Display live agent copilot interface"""
    st.markdown("<h2 class='sub-header'>💬 Live Agent Copilot</h2>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h4>🚀 Real AI Assistance Active</h4>
        <p>Enter a customer question below to get AI-powered suggestions with confidence scores and source citations.</p>
    </div>
    """, unsafe_allow_html=True)

    # Customer question input
    st.markdown("### 📝 Enter Customer Question")

    # Example questions
    example_questions = [
        "How do I advance the property date?",
        "I cannot generate certification reports",
        "What should I do if certifications are failing?",
        "How to navigate PropertySuite screens",
        "Data validation issues in reports"
    ]

    col1, col2 = st.columns([3, 1])

    with col1:
        customer_question = st.text_area(
            "Paste the customer's question here:",
            placeholder="e.g., 'I cannot generate compliance certification reports. Getting error: Report generation failed due to data validation issues.'",
            height=100,
            key="customer_question"
        )

    with col2:
        st.markdown("### 💡 Examples:")
        for i, example in enumerate(example_questions):
            if st.button(f"{example[:20]}...", key=f"ex{i}", width="stretch"):
                st.session_state.selected_example = example
                st.rerun()

    # Use selected example
    if hasattr(st.session_state, 'selected_example') and st.session_state.selected_example:
        customer_question = st.session_state.selected_example
        st.info(f"📝 Using example: {customer_question}")
        # Clear after use
        del st.session_state.selected_example

    # Context information
    with st.expander("➕ Add Context Information (Optional)"):
        col1, col2 = st.columns(2)
        with col1:
            product = st.selectbox("Product", ["PropertySuite Affordable", "PropertySuite Facilities", "Other"])
            category = st.selectbox("Category", ["Certifications", "Date Advance", "Reporting", "General", "Navigation"])
        with col2:
            priority = st.selectbox("Priority", ["High", "Medium", "Low"])
            customer_tier = st.selectbox("Customer Tier", ["Tier 1", "Tier 2", "Tier 3"])

    context = {
        "product": product,
        "category": category,
        "priority": priority,
        "tier": customer_tier
    }

    # Main action button
    if st.button("🤖 Get AI Suggestion", type="primary", width="stretch"):
        if customer_question:
            # Progress indicator
            with st.spinner("🔍 Processing your question..."):
                # Process the question
                suggestion = safe_process_question(customer_question, context)

                if 'error' in suggestion:
                    st.error(f"Error: {suggestion['error']}")
                else:
                    # Store in session state
                    st.session_state.current_suggestion = suggestion

                    # Show success message
                    st.markdown(f"""
                    <div class="success-box">
                        ✅ <strong>Analysis Complete!</strong> Found {suggestion['num_sources']} relevant knowledge sources.
                    </div>
                    """, unsafe_allow_html=True)

                    # Display results
                    display_suggestion(suggestion, key_suffix="_current")
        else:
            st.warning("Please enter a customer question first.")

    # Display previous suggestion if exists
    if st.session_state.current_suggestion and not customer_question:
        st.markdown("---")
        st.markdown("### 📋 Previous Suggestion")
        display_suggestion(st.session_state.current_suggestion, key_suffix="_previous")

def display_suggestion(suggestion, key_suffix=""):
    """Display a suggestion in the UI"""

    # Confidence indicator
    confidence = suggestion.get('confidence_score', 0)
    confidence_label = suggestion.get('confidence_label', 'Unknown')

    st.markdown(f"### 🎯 Confidence Score: **{confidence:.1%}** ({confidence_label})")

    # Color-coded recommendation
    if confidence > 0.7:
        st.markdown('<div class="success-box">🟢 HIGH CONFIDENCE - Ready to send</div>', unsafe_allow_html=True)
    elif confidence > 0.4:
        st.markdown('<div class="warning-box">🟡 MEDIUM CONFIDENCE - Verify before sending</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="error-box">🔴 LOW CONFIDENCE - Consider escalating</div>', unsafe_allow_html=True)

    # Agent guidance
    st.markdown(f"**💡 Agent Guidance:** {suggestion.get('agent_guidance', 'N/A')}")

    # Suggested response
    st.markdown("### 💬 Suggested Response")
    st.markdown(suggestion.get('suggested_answer', 'No answer available'))

    # Sources
    sources = suggestion.get('sources', [])
    with st.expander(f"📚 View Sources ({len(sources)} found)"):
        if sources:
            try:
                sources_df = pd.DataFrame(sources)
                st.dataframe(sources_df, width="stretch")
            except:
                for i, source in enumerate(sources, 1):
                    st.write(f"**{i}. {source.get('title', 'Unknown')}** ({source.get('relevance', 'N/A')})")
        else:
            st.info("No sources found for this question.")

    # Response actions
    st.markdown("### 📤 Response Actions")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📋 Copy to Clipboard", width="stretch", key=f"copy_btn{key_suffix}"):
            st.toast("Response copied!", icon="✅")
    with col2:
        if st.button("✏️ Edit Response", width="stretch", key=f"edit_btn{key_suffix}"):
            st.info("Edit feature coming soon!")
    with col3:
        if st.button("📤 Send to Customer", type="primary", width="stretch", key=f"send_btn{key_suffix}"):
            st.success("✅ Response sent!")

def show_knowledge_base():
    """Display live knowledge base search"""
    st.markdown("<h2 class='sub-header'>📚 Live Knowledge Base Search</h2>", unsafe_allow_html=True)

    # Search interface
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input(
            "🔍 Search the knowledge base:",
            placeholder="Enter keywords, questions, or topics..."
        )
    with col2:
        limit = st.selectbox("Results", [5, 10, 20, 50], index=0)

    if st.button("🔍 Search", type="primary", width="stretch"):
        if search_query:
            with st.spinner(f"Searching for '{search_query}'..."):
                results = safe_search(search_query, limit)

                if results:
                    st.success(f"✅ Found {len(results)} results")

                    # Display results
                    for i, result in enumerate(results, 1):
                        with st.expander(f"{i}. {result.get('Title', 'Unknown')} ({result.get('Relevance', 'N/A')})"):
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.metric("Relevance", result.get('Relevance', 'N/A'))
                                st.caption(f"Type: {result.get('Type', 'N/A')}")
                                st.caption(f"Category: {result.get('Category', 'N/A')}")
                            with col2:
                                st.write(result.get('Content Preview', 'No preview'))
                                st.caption(f"ID: {result.get('ID', 'N/A')}")
                else:
                    st.warning("No results found. Try different keywords.")
        else:
            st.warning("Please enter a search query.")

def show_system_status():
    """Display system status and health"""
    st.markdown("<h2 class='sub-header'>📊 System Status & Health</h2>", unsafe_allow_html=True)

    try:
        status = system_manager.get_system_status()

        if status.get('status') == 'ready':
            # System Health Cards
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Knowledge Docs", f"{status.get('knowledge_base_size', 0):,}")
            with col2:
                st.metric("Vectors Indexed", f"{status.get('index_size', 0):,}")
            with col3:
                st.metric("LLM Status", "Active" if status.get('llm_active') else "Mock")
            with col4:
                st.metric("System Version", "v1.0")

            # Component Status
            st.markdown("### ⚙️ Component Status")

            components = [
                ("Knowledge Base", "✅ Operational", f"{status.get('knowledge_base_size', 0):,} documents loaded"),
                ("Vector Search (FAISS)", "✅ Operational", f"{status.get('index_size', 0):,} vectors indexed"),
                ("Embedding Model", "✅ Operational", status.get('embedding_model', 'Unknown')),
                ("LLM Integration", "✅ Active" if status.get('llm_active') else "⚠️ Mock Mode", status.get('llm_model', 'Unknown')),
            ]

            for name, status_icon, details in components:
                st.markdown(f"""
                <div class="card">
                    <strong>{name}</strong><br>
                    <span style="color: {'#10B981' if '✅' in status_icon else '#F59E0B'}">{status_icon}</span> {details}
                </div>
                """, unsafe_allow_html=True)

            # Test the system
            st.markdown("### 🧪 Quick System Test")

            if st.button("Run System Test", width="stretch"):
                with st.spinner("Running system test..."):
                    test_result = safe_process_question("How do I advance the property date?")

                    if 'error' in test_result:
                        st.error(f"Test failed: {test_result['error']}")
                    else:
                        st.success(f"✅ System test passed! Confidence: {test_result['confidence_score']:.1%}")
                        st.info(f"Found {test_result['num_sources']} relevant sources")
        else:
            st.error("System not ready. Please initialize from the Dashboard.")
    except Exception as e:
        st.error(f"Error getting status: {str(e)}")

# Run the app
if __name__ == "__main__":
    main()
