import streamlit as st # Web interface framework
import pandas as pd # Data manipulation and CSV handling
import os # File and directory operations
from langchain_community.vectorstores import Chroma # Local vector database interface
from langchain_huggingface import HuggingFaceEmbeddings # Text-to-vector transformation model
from langchain_community.llms import Ollama # Local LLM (Llama-3) connector

# ====================================================================
# Logic Layer: The Semantic Agent
# ====================================================================

class SemanticIntentAgent:
    """Agent that uses vector memory to reclassify financial intents."""
    
    def __init__(self, db_path="agent_memory_db"):
        # Store the path to your existing vector database
        self.db_path = db_path 
        
        # Load the embedding model to ensure the math matches your stored data
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Connect to the persistent ChromaDB folder
        self.vectorstore = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings,
            collection_name="intent_memory"
        )
        
        # Initialize the local LLM instance
        self.llm = Ollama(model="llama3")

    def get_unique_categories(self):
        """Retrieves all distinct categories currently stored in the vector database."""
        # Access the underlying collection metadata to find unique labels
        data = self.vectorstore.get()
        if data and 'metadatas' in data:
            return list(set(m['output'] for m in data['metadatas']))
        return []

    def predict_intent(self, user_query):
        """Retrieves similar cases and asks Llama-3 for a final decision."""
        
        # Search: Find the top 3 most semantically similar examples
        similar_examples = self.vectorstore.similarity_search(user_query, k=3)
        
        # Context Building: Format retrieved examples into a string
        example_context = ""
        for doc in similar_examples:
            example_context += f"Example Input: {doc.page_content}\nCorrect Category: {doc.metadata['output']}\n\n"

        # Prompting: Construct the instruction set with historical context
        full_prompt = f"""
        You are a Financial Services Routing Agent.
        Use the following historical examples to categorize the new inquiry.

        ### Historical Context:
        {example_context}

        ### New Inquiry to Classify:
        {user_query}

        Instructions: Return ONLY the exact category name.
        """
        
        # Generation: Send the prompt to the LLM
        response = self.llm.invoke(full_prompt).strip()
        return response

    def update_memory(self, text, category):
        """Adds a new successful classification to the vector database."""
        self.vectorstore.add_texts(
            texts=[text],
            metadatas=[{"output": category}]
        )

# ====================================================================
# Processing Layer: Bulk Handling
# ====================================================================

class IntentProcessor:
    def __init__(self, agent):
        self.agent = agent 

    def run_reclassification(self, df, threshold):
        """Iterates through a dataframe to fix low-confidence errors."""
        try:
            results = []
            statuses = []

            for index, row in df.iterrows():
                # Trigger RAG prediction if confidence is below threshold
                if float(row['confidence_level']) < threshold:
                    new_intent = self.agent.predict_intent(row['customer_statement'])
                    results.append(new_intent)
                    statuses.append("Reclassified (Semantic RAG)")
                else:
                    results.append(row['department_routed'])
                    statuses.append("Original (High Confidence)")

            df['final_intent'] = results
            df['audit_status'] = statuses
            return df, "Success"
        except Exception as e:
            return None, str(e)

# ====================================================================
# Application Layer: Streamlit UI
# ====================================================================

def main():
    # Set the specific page title requested
    st.set_page_config(page_title="Customer Intent Re-Classification AI Agent", layout="wide")

    # Persistent Session State
    if 'agent' not in st.session_state:
        st.session_state.agent = SemanticIntentAgent()
    if 'processor' not in st.session_state:
        st.session_state.processor = IntentProcessor(st.session_state.agent)

    # Navigation
    st.sidebar.title("Agent Controls")
    nav = st.sidebar.radio("Select Task", ["Bulk Audit", "Memory Trainer", "Database Stats"])

    if nav == "Bulk Audit":
        st.title("Bulk Audit and Re-Classification")
        st.info("System evaluating vendor confidence against local semantic memory.")
        
        threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.60)
        uploaded_file = st.file_uploader("Upload Vendor Results (CSV)", type="csv")
        
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            if st.button("Start AI Re-classification"):
                with st.spinner("Analyzing narratives..."):
                    processed_df, msg = st.session_state.processor.run_reclassification(data, threshold)
                    if processed_df is not None:
                        st.success("Analysis complete.")
                        st.dataframe(processed_df)
                    else:
                        st.error(msg)

    elif nav == "Memory Trainer":
        st.title("Train Agent Memory")
        
        # Define the requested initial categories
        base_categories = [
            "password_reset", "transaction_query", "loan_inquiry", "fraud_report", 
            "credi_card_application", "balance_inquiry", "trading", "retirement", 
            "tax", "irrelevant"
        ]
        
        # Fetch existing categories from the DB and merge with base list
        db_categories = st.session_state.agent.get_unique_categories()
        full_cat_list = sorted(list(set(base_categories + db_categories)))

        with st.form("train_form"):
            new_text = st.text_area("Customer Statement:")
            
            # Category selection including existing and custom ones
            selected_cat = st.selectbox("Correct Category:", full_cat_list)
            
            # Allow user to add a brand new category
            custom_cat = st.text_input("OR Add New Category (Leave blank to use selection):")
            
            if st.form_submit_button("Inject into Memory"):
                final_cat = custom_cat.strip() if custom_cat else selected_cat
                if new_text and final_cat:
                    st.session_state.agent.update_memory(new_text, final_cat)
                    st.success(f"Example saved under category: {final_cat}")
                    st.rerun()

    elif nav == "Database Stats":
        st.title("Database Overview")
        count = st.session_state.agent.vectorstore._collection.count()
        st.metric("Total Learned Examples", count)
        
        st.subheader("Current Active Categories")
        st.write(st.session_state.agent.get_unique_categories())

if __name__ == "__main__":
    main()