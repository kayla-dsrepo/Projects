import streamlit as st
import pandas as pd
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# ====================================================================
# Logic Layer: The Semantic Agent
# ====================================================================

class SemanticIntentAgent:
    """Agent that uses vector memory to reclassify financial intents."""
    
    def __init__(self, db_path="agent_memory_db"):
        # Local folder where the agent's knowledge of past cases is stored
        self.db_path = db_path 
        
        # The engine that translates human sentences into a mathematical map
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Connection to the local database containing historical examples
        self.vectorstore = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings,
            collection_name="intent_memory"
        )
        
        # Connection to the local Llama-3 model for decision making
        self.llm = Ollama(model="llama3")

    def get_unique_categories(self):
        """Retrieves all distinct categories currently stored in the vector database."""
        data = self.vectorstore.get()
        if data and 'metadatas' in data:
            return list(set(m['output'] for m in data['metadatas']))
        return []

    def predict_intent(self, user_query):
            """
            Retrieves similar cases and generates a classification via Llama-3.
            Includes a fallback 'Mock Mode' for cloud deployment environments.
            """
            
            # 1. SEMANTIC RETRIEVAL: Find the 3 most similar past cases in ChromaDB
            # This part works both locally and in the cloud if the DB is uploaded.
            similar_examples = self.vectorstore.similarity_search(user_query, k=3)
            
            # 2. ATTEMPT AI GENERATION (Llama-3)
            try:
                # Prepare the reference context for the AI based on those past cases
                example_context = ""
                for doc in similar_examples:
                    example_context += f"Example Input: {doc.page_content}\nCorrect Category: {doc.metadata['output']}\n\n"

                # Directives for the AI to ensure consistent labeling
                full_prompt = f"""
                You are a Financial Services Routing Agent.
                Use the following historical examples to categorize the new inquiry.

                ### Historical Context:
                {example_context}

                ### New Inquiry to Classify:
                {user_query}

                ### Instructions:
                1. Return ONLY the exact category name.
                2. The category name should always be in lowercase, and words should be connected by an underscore.
                3. Any irrelevant query should be labeled exactly as 'irrelevant', not 'non-financial' or other labels.
                """
                
                # Try to invoke the local Llama-3 model via Ollama
                response = self.llm.invoke(full_prompt).strip().lower()
                return response

            except Exception:
                # 3. CLOUD FALLBACK (MOCK MODE)
                # If Ollama is unreachable, we use 'Nearest Neighbor' logic.
                # We return the category of the #1 most similar example in memory.
                # if similar_examples:
                #     # Get the metadata from the top result
                #     fallback_category = similar_examples[0].metadata['output']
                #     return fallback_category
                
                # # If the database is completely empty, return 'irrelevant'
                return "mock_mode_label"


    def update_memory(self, text, category):
        """Adds a new successful classification to the vector database."""
        # Sanitizes the label to ensure it follows the lowercase_underscore format
        clean_cat = category.strip().lower().replace(" ", "_")
        self.vectorstore.add_texts(
            texts=[text],
            metadatas=[{"output": clean_cat}]
        )

# ====================================================================
# Processing Layer: Bulk Handling
# ====================================================================

class IntentProcessor:
    def __init__(self, agent):
        self.agent = agent 

    def run_reclassification(self, df, threshold):
        """Processes a file to fix entries where the initial system was unsure."""
        try:
            results = []
            statuses = []

            for index, row in df.iterrows():
                # If the initial system's confidence is too low, our AI takes a second look
                if float(row['confidence_level']) < threshold:
                    new_intent = self.agent.predict_intent(row['customer_statement'])
                    results.append(new_intent)
                    statuses.append("Reclassified (AI Second Opinion)")
                else:
                    results.append(row['department_routed'])
                    statuses.append("Original (Confirmed)")

            df['final_intent'] = results
            df['audit_status'] = statuses
            return df, "Success"
        except Exception as e:
            return None, str(e)

# ====================================================================
# Application Layer: Streamlit UI
# ====================================================================

def main():
    st.set_page_config(page_title="Customer Intent Re-Classification AI Agent", layout="wide")

    if 'agent' not in st.session_state:
        st.session_state.agent = SemanticIntentAgent()
    if 'processor' not in st.session_state:
        st.session_state.processor = IntentProcessor(st.session_state.agent)

    # Sidebar Navigation
    st.sidebar.title("Navigation Menu")
    nav = st.sidebar.radio("Go to:", ["Welcome Page", "Bulk Audit Tool", "Train the AI", "System Health"])

    # --- WELCOME PAGE ---
    if nav == "Welcome Page":
        st.title("Welcome to the Intent Re-Classification AI Agent")
        
        st.markdown("""
        ### Introduction
        Automated customer service systems sometimes find it difficult to understand the specific details of a customer's request. This tool acts as a dedicated support layer that provides a second opinion on inquiries that the primary system found confusing.
        
        By using artificial intelligence, the agent looks at the overall meaning of a message rather than just searching for specific keywords. This ensures that customers are connected to the correct specialist as quickly as possible.
        
        ### How to Use This App
        * **Bulk Audit Tool:** Use this to upload and fix large lists of customer messages that were previously misrouted.
        * **Train the AI:** Use this to teach the system new examples of customer requests so it can learn from your expertise.
        * **System Health:** Check this section to see how much the agent has learned and which categories it currently understands.
        
        ---
        ### Support and Resources
        **Technical Support**
        If you have any questions or need assistance, please reach out to our team at: **kaylawang2112@gmail.com**
        
        **Project Documentation**
        For those interested in the underlying technology and project updates, visit our repository:
        [View Project on GitHub](https://github.com/kayla-dsrepo/Projects)
        """)

    # --- BULK AUDIT ---
    elif nav == "Bulk Audit Tool":
        st.title("Bulk Audit and Re-Classification")
        
        st.subheader("What this tool does")
        st.write("This tool processes files containing many customer requests. If the primary routing system was not confident in its decision, our AI will re-examine the request and provide a more accurate category.")
        
        st.subheader("Instructions")
        st.write("1. Set Sensitivity: Use the slider to choose when the AI should step in. For example, a setting of 0.6 means the AI will double-check any request where the original system was less than 60% sure.")
        st.write("2. Upload File: Select your CSV file. It must follow the specific format shown in the example below.")
        st.write("3. Run Audit: Click the button to start the analysis. You can download the corrected list once finished.")
        
        st.info("**Example File Format (First 2 Lines):**")
        st.code("""customer_statement,department_routed,confidence_level
"I forgot my login password and need a reset","password_reset",0.9
"Why was my $50 transfer to John Doe declined?","transaction_query",0.4""")
        
        st.divider()
        
        threshold = st.slider("AI Intervention Threshold", 0.0, 1.0, 0.6, step=0.1)
        uploaded_file = st.file_uploader("Upload Vendor Results (CSV)", type="csv")
        
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            if st.button("Start Re-Classification Process"):
                with st.spinner("The AI is currently analyzing customer narratives..."):
                    processed_df, msg = st.session_state.processor.run_reclassification(data, threshold)
                    if processed_df is not None:
                        st.success("Re-classification complete.")
                        st.dataframe(processed_df)
                        
                        csv = processed_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Corrected Results",
                            data=csv,
                            file_name='corrected_intents.csv',
                            mime='text/csv',
                        )
                    else:
                        st.error(msg)

    # --- MEMORY TRAINER ---
    elif nav == "Train the AI":
        st.title("Train the AI's Memory")
        
        st.subheader("What this tool does")
        st.write("You can help the AI get smarter by providing examples of how real customers talk. By saving these examples, you are teaching the system how to handle similar requests in the future.")
        
        st.subheader("Instructions")
        st.write("1. Enter a Message: Type in a message a customer might say, like 'I need to check my current mortgage rate'.")
        st.write("2. Select Category: Choose the correct department for this request, such as 'loan_inquiry'.")
        st.write("3. Save: Click the button to add this to the AI's permanent memory.")
        
        st.divider()

        base_categories = [
            "password_reset", "transaction_query", "loan_inquiry", "fraud_report", 
            "credi_card_application", "balance_inquiry", "trading", "retirement", 
            "tax", "irrelevant"
        ]
        
        db_categories = st.session_state.agent.get_unique_categories()
        full_cat_list = sorted(list(set(base_categories + db_categories)))

        with st.form("train_form"):
            new_text = st.text_area("Customer Message Example:")
            selected_cat = st.selectbox("Assign to Category:", full_cat_list)
            custom_cat = st.text_input("OR Create a New Category (example: insurance_claim):")
            
            if st.form_submit_button("Save to AI Memory"):
                final_cat = custom_cat.strip().lower().replace(" ", "_") if custom_cat else selected_cat
                if new_text and final_cat:
                    st.session_state.agent.update_memory(new_text, final_cat)
                    st.success(f"Successfully saved. The AI has now learned this example for the category: {final_cat}")
                    st.rerun()

    # --- DATABASE STATS ---
    elif nav == "System Health":
        st.title("System Knowledge Overview")
        
        st.subheader("What this tool does")
        st.write("This dashboard shows you how much the agent has learned so far. It counts the number of examples you have saved and lists the departments it is currently able to recognize.")
        
        st.divider()
        
        count = st.session_state.agent.vectorstore._collection.count()
        st.metric("Total Examples Learned", count, help="This is the number of 'reference cases' the AI uses to make decisions.")
        
        st.subheader("Recognized Departments")
        st.write("The AI is currently trained to understand requests for the following departments:")
        st.write(st.session_state.agent.get_unique_categories())

if __name__ == "__main__":
    main()