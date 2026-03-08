import streamlit as st
import pandas as pd
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.llms import Ollama

# ====================================================================
# Logic Layer: Semantic Router with Dynamic Few-Shot Learning
# ====================================================================

class SemanticFidelityRouter:
    """Handles classification using a vector-memory of past examples."""
    
    def __init__(self):
        # 1. Initialize local embedding model (runs on your CPU)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # 2. Initial "Seed" Examples (The Memory)
        self.initial_examples = [
            {"input": "I need to reset my password", "output": "Service"},
            {"input": "I want to buy 100 shares of apple", "output": "Trading"},
            {"input": "How much can I put in my 401k?", "output": "Retirement"},
            {"input": "Where is my 1099 form?", "output": "Tax"},
            {"input": "What is the weather like in Boston?", "output": "Irrelevant"}
        ]
        
        # 3. Initialize Vector Store (Persistent Memory)
        self.vectorstore = Chroma(
            persist_directory="./fidelity_vector_db",
            embedding_function=self.embeddings,
            collection_name="intent_memory"
        )
        
        # Seed the DB if it's empty
        if self.vectorstore._collection.count() == 0:
            self._seed_db()

        # 4. Setup Local LLM (Llama 3 via Ollama)
        self.llm = Ollama(model="llama3")

    def _seed_db(self):
        for ex in self.initial_examples:
            self.add_new_example(ex['input'], ex['output'])

    def add_new_example(self, text, department):
        """Saves a new correct classification to the vector memory."""
        self.vectorstore.add_texts(
            texts=[text],
            metadatas=[{"output": department}]
        )

    def predict_department(self, query):
        """Retrieves similar examples and uses Few-Shot prompting to classify."""
        
        # Retrieve the top 3 most similar examples from memory
        similar_docs = self.vectorstore.similarity_search(query, k=3)
        
        # Format examples for the prompt
        example_context = ""
        for doc in similar_docs:
            example_context += f"Input: {doc.page_content}\nOutput: {doc.metadata['output']}\n\n"

        prompt = f"""
        You are a Fidelity Investments routing specialist. 
        Classify the input into: Trading, Retirement, Service, Tax, or Irrelevant.

        ### Past Examples for Context:
        {example_context}

        ### New Task:
        Input: {query}
        Output: (Return ONLY the department name)
        """
        
        response = self.llm.invoke(prompt).strip()
        # Clean up the response to ensure it matches your categories
        categories = ["Trading", "Retirement", "Service", "Tax", "Irrelevant"]
        for cat in categories:
            if cat.lower() in response.lower():
                return cat
        return "Irrelevant"

# ====================================================================
# Processor Layer
# ====================================================================

class DataFrameProcessor:
    def __init__(self, router):
        self.router = router

    def process_dataframe(self, df, threshold):
        try:
            required_cols = ['customer_statement', 'department_routed', 'confidence_level']
            if not all(col in df.columns for col in required_cols):
                return None, "Error: Missing required columns."

            final_depts = []
            status = []

            for _, row in df.iterrows():
                conf = float(row['confidence_level'])
                if conf < threshold:
                    # Logic Recovery via Semantic RAG
                    prediction = self.router.predict_department(row['customer_statement'])
                    final_depts.append(prediction)
                    status.append(f"Reclassified (Semantic Memory)")
                else:
                    final_depts.append(row['department_routed'])
                    status.append("Original")

            df['final_classification'] = final_depts
            df['processing_status'] = status
            return df, "success"
        except Exception as e:
            return None, str(e)

# ====================================================================
# Streamlit UI
# ====================================================================

def main():
    st.set_page_config(page_title="Semantic Intent Re-router", layout="wide")

    if 'router' not in st.session_state:
        st.session_state.router = SemanticFidelityRouter()
    
    if 'processor' not in st.session_state:
        st.session_state.processor = DataFrameProcessor(st.session_state.router)

    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Go to:", ["1. Bulk Reclassification", "2. Train the Memory", "3. About"])

    if option == "1. Bulk Reclassification":
        st.title("📂 Semantic Logic Recovery")
        threshold = st.slider("Low-Confidence Threshold", 0.0, 1.0, 0.6)
        
        uploaded_file = st.file_uploader("Upload Vendor CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if st.button("Run Semantic Re-router"):
                with st.spinner("Retrieving similar cases from memory and re-classifying..."):
                    result, msg = st.session_state.processor.process_dataframe(df, threshold)
                    if result is not None:
                        st.dataframe(result)
                    else:
                        st.error(msg)

    elif option == "2. Train the Memory":
        st.title("🧠 Supervisor Training Interface")
        st.write("Instead of keywords, provide a full sentence example to improve the model's 'Memory'.")
        
        with st.form("training_form"):
            example_text = st.text_input("Customer Statement Example:")
            correct_dept = st.selectbox("Correct Department:", ["Trading", "Retirement", "Service", "Tax", "Irrelevant"])
            submitted = st.form_submit_button("Add to Semantic Memory")
            
            if submitted and example_text:
                st.session_state.router.add_new_example(example_text, correct_dept)
                st.success(f"Added to Vector Store! The system now knows that '{example_text}' belongs to {correct_dept}.")

    elif option == "3. About":
        st.title("LangChain-Powered Intent Routing")
        st.info("This version uses **Vector Search (ChromaDB)** and **Few-Shot Prompting (Llama 3)** to recover logic from low-confidence predictions.")

if __name__ == "__main__":
    main()