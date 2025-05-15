import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os
import time
import base64
from pathlib import Path
from predictor import COMPARTMENT_CLASSES, ProtGPSPredictor

# Configure the page
st.set_page_config(
    page_title="ProtGPS Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize predictor
@st.cache_resource
def get_predictor():
    predictor = ProtGPSPredictor()
    predictor.load_model()
    return predictor

# Function to generate download link
def get_download_link(df, filename, text):
    """Generate a download link for a dataframe"""
    excel_file = df.to_excel(index=False)
    b64 = base64.b64encode(excel_file).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Function to create a heatmap
def create_heatmap(df):
    score_columns = [col for col in df.columns if col.endswith('_Score')]
    score_data = df[score_columns].values
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.3 + 2))
    
    # Create the heatmap
    im = ax.imshow(score_data, cmap='viridis', aspect='auto')
    
    # Set ticks and labels
    compartment_labels = [col.replace('_Score', '') for col in score_columns]
    ax.set_xticks(np.arange(len(compartment_labels)))
    ax.set_yticks(np.arange(len(df)))
    
    # Set the labels
    ax.set_xticklabels(compartment_labels, rotation=45, ha="right")
    ax.set_yticklabels(df['Label'])
    
    # Add a colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Score", rotation=-90, va="bottom")
    
    # Add title
    plt.title("Condensate Localization Prediction Scores")
    plt.tight_layout()
    
    return fig

# Main function
def main():
    st.title("🧬 ProtGPS Predictor")
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.markdown("""
    **ProtGPS Predictor** is a tool developed by the Whitehead Institute for predicting protein condensate localization.
    
    This Streamlit app allows you to upload protein sequences in FASTA format and get predictions for where these proteins may localize within cells.
    """)
    
    # Load predictor (cached)
    try:
        predictor = get_predictor()
        model_status = "✅ Model loaded successfully"
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        model_status = "❌ Model loading failed"
    
    st.sidebar.text(model_status)
    
    # Configuration options
    st.sidebar.header("Settings")
    batch_size = st.sidebar.slider("Batch Size", min_value=1, max_value=16, value=4, 
                                  help="Number of sequences to process at once")
    
    # Main area
    st.markdown("""
    Upload a FASTA file containing protein sequences to predict their cellular condensate localization.
    
    The prediction model will assign scores to each sequence indicating its likelihood of localizing to different cellular compartments.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload FASTA file", type=["fasta", "fa", "txt"])
    
    # Example file option
    use_example = st.checkbox("Use example sequences")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Prediction", "Results Explanation", "Input Format"])
    
    with tab1:
        if uploaded_file is not None or use_example:
            # Process the file
            with tempfile.TemporaryDirectory() as temp_dir:
                # Determine which file to use
                if use_example:
                    input_path = os.path.join(Path(__file__).parent, "data", "example.fasta")
                else:
                    # Save uploaded file to temp directory
                    input_path = os.path.join(temp_dir, "input.fasta")
                    with open(input_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Output path
                output_path = os.path.join(temp_dir, "results.xlsx")
                
                # Display file info
                try:
                    with open(input_path, "r") as f:
                        sequence_count = sum(1 for line in f if line.startswith(">"))
                    st.write(f"File contains {sequence_count} sequences")
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    sequence_count = 0
                
                # Show prediction button if sequences found
                if sequence_count > 0:
                    if st.button("Run Prediction"):
                        # Setup progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        def update_progress(progress):
                            progress_bar.progress(progress)
                            status_text.text(f"Processing... {progress}%")
                        
                        try:
                            # Run prediction
                            status_text.text("Starting prediction...")
                            start_time = time.time()
                            
                            # Run the prediction
                            results_df = predictor.predict_from_file(
                                input_path, 
                                output_path, 
                                batch_size=batch_size,
                                progress_callback=update_progress
                            )
                            
                            # Calculate runtime
                            runtime = time.time() - start_time
                            status_text.text(f"Prediction completed in {runtime:.2f} seconds!")
                            
                            # Display results
                            st.subheader("Results")
                            st.dataframe(results_df)
                            
                            # Create visualization
                            st.subheader("Visualization")
                            fig = create_heatmap(results_df)
                            st.pyplot(fig)
                            
                            # Download link
                            st.markdown("### Download Results")
                            st.markdown(f"Download the full results as an Excel file:")
                            
                            # Create download button
                            buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
                            results_df.to_excel(buffer, index=False)
                            buffer.close()
                            
                            with open(buffer.name, "rb") as f:
                                bytes_data = f.read()
                            
                            st.download_button(
                                label="Download Excel File",
                                data=bytes_data,
                                file_name="protgps_results.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                            
                        except Exception as e:
                            st.error(f"Error during prediction: {str(e)}")
    
    with tab2:
        st.subheader("Understanding Prediction Results")
        st.markdown("""
        The prediction results include scores for each protein's likelihood of localizing to different cellular compartments.
        Scores range from 0.0 to 1.0, with higher values indicating stronger prediction confidence.
        """)
        
        # Get compartment descriptions
        descriptions = predictor.get_compartment_descriptions() if 'predictor' in locals() else {}
        
        # Display descriptions in a table
        st.write("Compartment Descriptions:")
        desc_df = pd.DataFrame([
            {"Compartment": comp, "Description": descriptions.get(comp, "")}
            for comp in COMPARTMENT_CLASSES
        ])
        st.table(desc_df)
        
        st.markdown("""
        ### Interpreting the Results
        
        - **High scores (>0.7)** suggest strong confidence that the protein localizes to that compartment
        - **Medium scores (0.3-0.7)** indicate moderate confidence
        - **Low scores (<0.3)** suggest the protein is unlikely to localize to that compartment
        
        Proteins may score highly for multiple compartments, which often indicates that they shuttle between these locations or have multiple functions.
        """)
    
    with tab3:
        st.subheader("FASTA Format Requirements")
        st.markdown("""
        Your FASTA file should follow this format:
        
        ```
        >gene|organism|uniprot_id|mutation_status
        SEQUENCE
        >gene2|organism2|uniprot_id2|mutation_status2
        SEQUENCE2
        ```
        
        Example:
        ```
        >FUS|human|P35637|WT
        MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQS
        >TDP43|human|Q13148|WT
        MSEYIRVTEDENDEPIEIPSEDDGTVLLSTVTAQFPGACGLRYRNPVSQCMRGVRLVEGILHAPDAGWGNLVYVVNYPKDNKRKMDETDASSAVKVKRAVQKTSDLIVLGLPWKTTEQDLKEYFSTFGEVLMVQVKKDLKTGHSKGFGFVRFTEYETQVKVMSQRHMIDGRWCDCKLPNSKQSQDEPLRSRKVFVGRCTEDMTEDELREFFSQYGDVMDVFIPKPFRAFAFVTFADDQIAQSLCGEDLIIKGISVHISNAEPKHNSNRQLERSGRFGGNPGGFGNQGGFGNSRGGGAGLGNNQGSNMGGGMNFGAFSINPAMMAAAQAALQSSWGMMGMLASQQNQSGPSGNNQNQGNMQREPNQAFGSGNNSYSGSNSGAAIGWGSASNAGSGSGFNGGFGSSMDSKSSGWGM
        ```
        """)
        
        st.markdown("""
        **Important Notes:**
        
        1. Make sure your sequences don't contain spaces or unexpected characters
        2. Each sequence header should start with ">"
        3. For best results, use the format shown above
        4. If your header doesn't match this format, the predictor will still work but will use the entire header as the label
        """)

if __name__ == "__main__":
    main()