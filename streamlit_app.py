import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Your EC2 public IP — the API we deployed in Phase 6
API_URL = "http://3.16.1.218:8000"

# Page configuration
st.set_page_config(
    page_title="Financial Document Q&A",
    page_icon="📊",
    layout="centered"
)

# Header
st.title("📊 Financial Document Q&A")
st.markdown("""
Ask questions about Apple's 2025 Annual Report (10-K).
Answers are grounded in the actual document with page citations.
""")

st.divider()

# Document info
with st.expander("📁 Documents in the knowledge base"):
    st.markdown("""
    - **Apple 10-K 2025** — Annual Report filed with the SEC (Oct 31, 2025)
    """)

st.divider()

# Question input
question = st.text_input(
    "Your question:",
    placeholder="e.g. What were Apple's total net sales in 2025?",
    help="Ask anything about the financial documents above"
)

# Ask button
if st.button("Ask", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Searching documents and generating answer..."):
            try:
                # Call our FastAPI endpoint on EC2
                response = requests.post(
                    f"{API_URL}/query",
                    json={"question": question},
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()

                    # Display the answer
                    st.markdown("### Answer")
                    st.markdown(result["answer"])

                    # Display sources
                    st.markdown("### Sources")
                    for i, source in enumerate(result["sources"]):
                        similarity_pct = round(source["similarity_score"] * 100, 1)
                        st.markdown(
                            f"**Source {i+1}:** {source['source']} — "
                            f"Page {int(source['page_number'])} "
                            f"*(relevance: {similarity_pct}%)*"
                        )
                else:
                    st.error(f"API error: {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API. Make sure EC2 is running.")
            except requests.exceptions.Timeout:
                st.error("Request timed out. The API may be busy.")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

st.divider()

# Footer
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
Built with FastAPI · Pinecone · OpenAI GPT-4o-mini · Amazon EC2<br>
Answers are grounded in retrieved document chunks only.
</div>
""", unsafe_allow_html=True)