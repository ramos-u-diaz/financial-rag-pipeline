import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Switch between local and EC2 depending on whether EC2 is running
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Financial Document Q&A",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Financial Document Q&A")
st.markdown("""
Ask questions about SEC 10-K filings from Apple, Allstate, and Progressive.
Answers are grounded in the actual documents with page citations.
""")

st.divider()

with st.expander("📁 Documents in the knowledge base"):
    st.markdown("""
    - **Apple 10-K 2025** — Annual Report filed with the SEC (Oct 2025)
    - **Allstate 10-K 2025** — Annual Report filed with the SEC (Feb 2026)
    - **Progressive 10-K 2025** — Annual Report filed with the SEC (Feb 2026)
    """)

st.divider()

# Company filter dropdown
company_options = ["All Companies", "Apple", "Allstate", "Progressive"]
selected_company = st.selectbox(
    "Filter by company:",
    options=company_options,
    help="Restrict answers to a specific company, or search across all documents"
)

# Convert "All Companies" to None for the API
company_filter = None if selected_company == "All Companies" else selected_company

# Question input
question = st.text_input(
    "Your question:",
    placeholder="e.g. What was net income in 2025?",
    help="Ask anything about the financial documents above"
)

if st.button("Ask", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Searching documents and generating answer..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={
                        "question": question,
                        "company": company_filter    # ← None or company name
                    },
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()

                    # Show which filter was applied
                    if result["company_filter"]:
                        st.info(f"Filtered to: {result['company_filter']} documents only")

                    st.markdown("### Answer")
                    st.markdown(result["answer"])

                    st.markdown("### Sources")
                    for i, source in enumerate(result["sources"]):
                        similarity_pct = round(source["similarity_score"] * 100, 1)
                        st.markdown(
                            f"**Source {i+1}:** {source['company']} — "
                            f"{source['source']} — "
                            f"Page {int(source['page_number'])} "
                            f"*(relevance: {similarity_pct}%)*"
                        )
                else:
                    st.error(f"API error: {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API. Make sure the server is running.")
            except requests.exceptions.Timeout:
                st.error("Request timed out. The API may be busy.")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

st.divider()

st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
Built with FastAPI · Pinecone · OpenAI GPT-4o-mini · Amazon Bedrock · Amazon EC2<br>
Answers are grounded in retrieved document chunks only.
</div>
""", unsafe_allow_html=True)