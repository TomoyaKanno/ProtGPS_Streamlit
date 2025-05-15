import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os
import time
import base64
import requests
import re
import html
from pathlib import Path
from io import StringIO

BATCH_SIZE = 4

# Import this from your existing predictor module
# Replace with your actual imports
try:
    from predictor import COMPARTMENT_CLASSES, ProtGPSPredictor
except ImportError:
    # For demonstration if predictor module isn't available
    COMPARTMENT_CLASSES = ["Nucleus", "Cytoplasm", "ER", "Mitochondria", "PML_Body", "Stress_Granule"]
    
    class ProtGPSPredictor:
        def __init__(self):
            pass
            
        def load_model(self):
            pass
            
        def predict_from_file(self, input_path, output_path, batch_size=4, progress_callback=None):
            # Mock function for demonstration
            data = {
                'Label': ['Protein1', 'Protein2'],
                'Sequence': ['MASNDYTQQATQSYGAYPTQPGQG', 'MSEYIRVTEDENDEPIEIPSEDDG']
            }
            
            # Add mock scores for each compartment
            for compartment in COMPARTMENT_CLASSES:
                data[f'{compartment}_Score'] = np.random.random(len(data['Label']))
                
            # Create mock progress updates
            if progress_callback:
                for i in range(0, 101, 10):
                    time.sleep(0.1)  # Simulate processing time
                    progress_callback(i)
                    
            return pd.DataFrame(data)
            
        def get_compartment_descriptions(self):
            return {comp: f"Description for {comp}" for comp in COMPARTMENT_CLASSES}


# Configure the page
st.set_page_config(
    page_title="Super ProtGPS Predictor",
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

# Function to fetch data from UniProt
def fetch_uniprot_data(accession):
    url = f"https://rest.uniprot.org/uniprotkb/{accession}?fields=accession%2Cprotein_name%2C%20sequence%2C%20organism_name%2C%20gene_primary"
    response = requests.get(url)
    return response.json()

# Function to format sequence entry
def format_sequence_entry(data, mutation=None):
    protein_name = data["proteinDescription"]["recommendedName"]["fullName"]["value"]
    organism = data["organism"]["scientificName"]
    
    # Add mutation information or [WT] right after UniProt ID
    mutation_info = f"[{mutation}]" if mutation else "[WT]"
    comment = f"# UniProt {data['primaryAccession']} {mutation_info} - {protein_name} ({organism})"
    
    sequence = data["sequence"]["value"]
    return comment, sequence

# Function to apply mutation
def apply_mutation(sequence, mutation_code):
    """Apply a mutation to a protein sequence"""
    # Regular expression to match mutation codes like P30R or P30TER
    pattern = r"([A-Z])(\d+)([A-Z]{1,3})"
    match = re.match(pattern, mutation_code)
    
    if not match:
        return None, "Invalid mutation format. Use format like 'P30R' or 'P30TER'."
    
    orig_aa, position, new_aa = match.groups()
    position = int(position) - 1  # Convert to 0-based index
    
    # Validate position
    if position < 0 or position >= len(sequence):
        return None, f"Position out of range. Sequence length is {len(sequence)}."
    
    # Validate original amino acid
    if sequence[position] != orig_aa:
        return None, f"Original amino acid mismatch. Expected {orig_aa} but found {sequence[position]} at position {int(position) + 1}."
    
    # Apply mutation
    if new_aa == "TER":
        # Termination - truncate the sequence
        mutated_sequence = sequence[:position]
    else:
        # Amino acid substitution
        mutated_sequence = sequence[:position] + new_aa + sequence[position + 1:]
    
    return mutated_sequence, None

# Function to format FASTA entry
def format_fasta_entry(data, sequence, mutation_code=None):
    """Format a sequence entry in FASTA format"""
    # Extract information for FASTA header
    gene = data["genes"][0]["geneName"]["value"] if "genes" in data and data["genes"] else "Unknown"
    organism = data["organism"]["commonName"] if "commonName" in data["organism"] else data["organism"]["scientificName"]
    uniprot_id = data["primaryAccession"]
    mutation_info = mutation_code if mutation_code else "WT"
    
    # Create FASTA header: >gene|organism|uniprot_id|mutation_status
    fasta_header = f">{gene}|{organism}|{uniprot_id}|{mutation_info}"
    
    # Format sequence with line breaks every 60 characters (standard FASTA format)
    formatted_sequence = ""
    for i in range(0, len(sequence), 60):
        formatted_sequence += sequence[i:i+60] + "\n"
    
    return f"{fasta_header}\n{formatted_sequence.strip()}"

# Function to create a heatmap for prediction results
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

# Function to combine all FASTA sequences into a single string
def combine_fasta_sequences(fasta_sequences):
    return "\n".join(fasta_sequences)

# Initialize session state for both tabs
if 'lib_sequences' not in st.session_state:
    st.session_state.lib_sequences = []
if 'lib_labels' not in st.session_state:
    st.session_state.lib_labels = []
if 'lib_current_data' not in st.session_state:
    st.session_state.lib_current_data = None
if 'lib_mutated_sequence' not in st.session_state:
    st.session_state.lib_mutated_sequence = None
if 'lib_mutation_code' not in st.session_state:
    st.session_state.lib_mutation_code = None
if 'lib_fasta_sequences' not in st.session_state:
    st.session_state.lib_fasta_sequences = []
if 'transferred_fasta' not in st.session_state:
    st.session_state.transferred_fasta = None

# Main app
def main():
    st.title("🧬 Super ProtGPS Predictor")
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.markdown("""
    **Super ProtGPS Predictor** combines sequence collection and prediction capabilities:
    
    1. **Library & Mutation Tab**: Collect and modify protein sequences from UniProt
    2. **Prediction Tab**: Predict protein condensate localization
    
    This tool was developed by the Whitehead Institute.
    """)
    
    # Load predictor (cached)
    try:
        predictor = get_predictor()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Create main tabs
    tab1, tab2 = st.tabs(["Library & Mutation", "Prediction"])
    
    # Library & Mutation Tab
    with tab1:
        st.header("UniProt Sequence Collector")
        
        # Input section with two side-by-side input boxes
        col1, col2 = st.columns(2)
        with col1:
            accession = st.text_input("Enter UniProt Accession ID", value="Q9Y5B6", key="lib_accession")
        with col2:
            mutation_code = st.text_input("Enter mutation (Optional)", placeholder="e.g., P30R or P30TER", key="lib_mutation_input")
        
        # Fetch and Format button - using full width
        if st.button("Fetch and Format", key="lib_fetch_button"):
            data = fetch_uniprot_data(accession)
            st.session_state.lib_current_data = data
            
            # Get the original sequence data
            original_comment, original_sequence = format_sequence_entry(data)
            
            # If mutation is provided, apply it
            if mutation_code:
                st.session_state.lib_mutation_code = mutation_code
                mutated_sequence, error = apply_mutation(original_sequence, mutation_code)
                
                if error:
                    st.error(error)
                    # Show the original sequence if there's an error
                    st.code(f'{original_comment}\n"{original_sequence}"', language="python")
                    st.session_state.lib_mutated_sequence = None
                else:
                    st.session_state.lib_mutated_sequence = mutated_sequence
                    
                    # Extract mutation position and check if it's a termination
                    pattern = r"([A-Z])(\d+)([A-Z]{1,3})"
                    match = re.match(pattern, mutation_code)
                    orig_aa, position, new_aa = match.groups()
                    is_termination = (new_aa == "TER")
                    pos_index = int(position) - 1  # 0-based index
                    
                    # Create a better mutation preview
                    st.subheader("Mutation Preview")
                    
                    # Create containers for side-by-side display
                    preview_col1, preview_col2 = st.columns(2)
                    
                    # Original sequence display with highlight
                    with preview_col1:
                        st.markdown("**Original:**")
                        st.text(f"Position: {position}")
                        
                        # Format sequence with highlighted position
                        formatted_orig = (
                            f'<div style="font-family:monospace; max-width:100%; overflow-x:auto; '
                            f'border:1px solid #ccc; padding:8px; border-radius:4px;">'
                            f'<div style="word-wrap:break-word; white-space:pre-wrap; width:100%;">'
                            f'{html.escape(original_sequence[:pos_index])}'
                            f'<span style="background-color:#ff6b6b;color:white;padding:0 2px;">{html.escape(original_sequence[pos_index])}</span>'
                            f'{html.escape(original_sequence[pos_index + 1:])}'
                            f'</div></div>'
                        )
                        st.markdown(formatted_orig, unsafe_allow_html=True)
                    
                    # Mutated sequence display with highlight
                    with preview_col2:
                        st.markdown("**Mutated:**")
                        st.text(f"Mutation: {mutation_code}")
                        
                        if is_termination:
                            # For termination, show truncated sequence with marker
                            formatted_mut = (
                                f'<div style="font-family:monospace; max-width:100%; overflow-x:auto; '
                                f'border:1px solid #ccc; padding:8px; border-radius:4px;">'
                                f'<div style="word-wrap:break-word; white-space:pre-wrap; width:100%;">'
                                f'{html.escape(mutated_sequence)}'
                                f'<span style="background-color:#ff6b6b;color:white;padding:0 2px;">■</span>'
                                f'</div></div>'
                            )
                        else:
                            # Format sequence with highlighted mutation
                            formatted_mut = (
                                f'<div style="font-family:monospace; max-width:100%; overflow-x:auto; '
                                f'border:1px solid #ccc; padding:8px; border-radius:4px;">'
                                f'<div style="word-wrap:break-word; white-space:pre-wrap; width:100%;">'
                                f'{html.escape(mutated_sequence[:pos_index])}'
                                f'<span style="background-color:#ff6b6b;color:white;padding:0 2px;">{html.escape(mutated_sequence[pos_index])}</span>'
                                f'{html.escape(mutated_sequence[pos_index + 1:])}'
                                f'</div></div>'
                            )
                        st.markdown(formatted_mut, unsafe_allow_html=True)
                    
                    # Show full mutated sequence
                    st.subheader("Mutated Sequence")
                    comment_with_mutation, _ = format_sequence_entry(data, mutation_code)
                    st.code(f'{comment_with_mutation}\n"{mutated_sequence}"', language="python")
            else:
                # If no mutation is provided, show the original sequence
                st.session_state.lib_mutation_code = None
                st.session_state.lib_mutated_sequence = None
                st.code(f'{original_comment}\n"{original_sequence}"', language="python")
        
        # Add to List button
        if st.button("Add to List", key="lib_add_button"):
            if st.session_state.lib_current_data is not None:
                # Determine if we're adding the original or mutated sequence
                if st.session_state.lib_mutated_sequence is not None:
                    comment, _ = format_sequence_entry(st.session_state.lib_current_data, st.session_state.lib_mutation_code)
                    sequence = st.session_state.lib_mutated_sequence
                else:
                    comment, sequence = format_sequence_entry(st.session_state.lib_current_data)
                
                formatted_entry = f'{comment}\n"{sequence}"'
                
                # Create reordered label with organism first, protein name second, and mutation/WT last
                data = st.session_state.lib_current_data
                protein_name = data["proteinDescription"]["recommendedName"]["fullName"]["value"]
                organism = data["organism"]["commonName"] if "commonName" in data["organism"] else data["organism"]["scientificName"]
                gene = data["genes"][0]["geneName"]["value"] if "genes" in data and data["genes"] else "Unknown"
                    
                mutation_info = f"[{st.session_state.lib_mutation_code}]" if st.session_state.lib_mutation_code else "[WT]"
                reordered_label = f"{organism}, {gene}, {mutation_info}"
                
                # Create FASTA formatted entry
                fasta_entry = format_fasta_entry(
                    data, 
                    sequence, 
                    st.session_state.lib_mutation_code
                )
                
                if formatted_entry not in st.session_state.lib_sequences:
                    st.session_state.lib_sequences.append(formatted_entry)
                    st.session_state.lib_labels.append(reordered_label)
                    st.session_state.lib_fasta_sequences.append(fasta_entry)
                    st.success("Sequence added to list!")
                else:
                    st.warning("This sequence is already in the list!")
            else:
                st.warning("Please fetch a sequence first!")
        
        # Display the collected sequences and labels side by side
        if st.session_state.lib_sequences:
            st.subheader("Collected Data")
            
            # Create two columns for side-by-side display
            col1, col2 = st.columns(2)
            
            # First column: Sequences
            with col1:
                st.markdown("### Sequences")
                formatted_sequences = "sequences = [\n    " + ",\n    ".join(st.session_state.lib_sequences) + "\n]"
                st.code(formatted_sequences, language="python")
            
            # Second column: Labels
            with col2:
                st.markdown("### Labels")
                formatted_labels = "labels = [\n    " + ",\n    ".join([f'"{label}"' for label in st.session_state.lib_labels]) + "\n]"
                st.code(formatted_labels, language="python")
            
            # Add export controls
            st.markdown("### Actions")
            export_col1, export_col2, export_col3 = st.columns(3)
            
            with export_col1:
                # Clear Lists button
                if st.button("Clear Lists", key="lib_clear_button"):
                    st.session_state.lib_sequences = []
                    st.session_state.lib_labels = []
                    st.session_state.lib_fasta_sequences = []
                    st.success("Lists cleared!")
            
            with export_col2:
                # Export to FASTA button
                if st.button("Export to FASTA", key="lib_export_button"):
                    # Join all FASTA entries
                    fasta_content = "\n".join(st.session_state.lib_fasta_sequences)
                    
                    # Create download button
                    st.download_button(
                        label="Download FASTA File",
                        data=fasta_content,
                        file_name="sequences.fasta",
                        mime="text/plain",
                        key="lib_download_button"
                    )
            
            with export_col3:
                # Send to Prediction button - key addition for integration
                if st.button("Send to Prediction", key="lib_send_button"):
                    if st.session_state.lib_fasta_sequences:
                        # Combine all sequences into a single FASTA string
                        fasta_content = combine_fasta_sequences(st.session_state.lib_fasta_sequences)
                        
                        # Store in session state for the Prediction tab
                        st.session_state.transferred_fasta = fasta_content
                        
                        # Notify user
                        st.success(f"Sent {len(st.session_state.lib_fasta_sequences)} sequences to Prediction tab!")
                        
                        # Suggest switching tabs
                        st.info("Please switch to the Prediction tab to analyze these sequences.")
                    else:
                        st.warning("No sequences available to send to Prediction.")

    # Prediction Tab
    with tab2:
        st.header("ProtGPS Prediction")
        
        # Check if sequences were transferred from Library tab
        has_transferred_sequences = st.session_state.transferred_fasta is not None
        
        # Show message if sequences were transferred
        if has_transferred_sequences:
            st.success(f"Sequences received from Library tab!")
            
            # Option to preview transferred sequences
            if st.checkbox("Preview transferred sequences", value=False, key="pred_preview_check"):
                st.code(st.session_state.transferred_fasta, language="")
            
            # Option to clear transferred sequences
            if st.button("Clear transferred sequences", key="pred_clear_transferred"):
                st.session_state.transferred_fasta = None
                st.experimental_rerun()
        
        st.markdown("""
        Upload a FASTA file containing protein sequences or use sequences from the Library tab 
        to predict their cellular condensate localization.
        
        The prediction model will assign scores to each sequence indicating its likelihood 
        of localizing to different cellular compartments.
        """)
        
        # File uploader (still available for backward compatibility)
        uploaded_file = st.file_uploader("Upload FASTA file", type=["fasta", "fa", "txt"], key="pred_file_upload")
        
        # Example file option
        use_example = st.checkbox("Use example sequences", key="pred_use_example")
        
        # Create tabs for the Prediction section
        pred_tab1, pred_tab2, pred_tab3 = st.tabs(["Prediction", "Results Explanation", "Input Format"])
        
        with pred_tab1:
            # Determine which source to use for prediction
            has_input = uploaded_file is not None or use_example or has_transferred_sequences
            
            if has_input:
                # Process the input
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Determine which input source to use
                    if has_transferred_sequences:
                        # Use transferred sequences from Library tab
                        input_path = os.path.join(temp_dir, "transferred.fasta")
                        with open(input_path, "w") as f:
                            f.write(st.session_state.transferred_fasta)
                        input_source = "transferred sequences"
                    elif use_example:
                        # Use example file
                        try:
                            example_path = os.path.join(Path(__file__).parent, "data", "example.fasta")
                            input_path = example_path
                            input_source = "example file"
                        except:
                            # If example file doesn't exist, create a mock one
                            input_path = os.path.join(temp_dir, "example.fasta")
                            with open(input_path, "w") as f:
                                f.write(">Example1|human|P12345|WT\nMASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQS\n>Example2|human|P67890|WT\nMSEYIRVTEDENDEPIEIPSEDDGTVLLSTVTAQFPGACGLRYRNPVSQCMRGVRLVEGIL")
                            input_source = "mock example"
                    else:
                        # Use uploaded file
                        input_path = os.path.join(temp_dir, "input.fasta")
                        with open(input_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        input_source = "uploaded file"
                    
                    # Output path
                    output_path = os.path.join(temp_dir, "results.xlsx")
                    
                    # Display file info
                    try:
                        with open(input_path, "r") as f:
                            sequence_count = sum(1 for line in f if line.startswith(">"))
                        st.write(f"File contains {sequence_count} sequences from {input_source}")
                    except Exception as e:
                        st.error(f"Error reading file: {str(e)}")
                        sequence_count = 0
                    
                    # Show prediction button if sequences found
                    if sequence_count > 0:
                        if st.button("Run Prediction", key="pred_run_button"):
                            # Setup progress tracking
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            def update_progress(progress):
                                progress_bar.progress(progress)
                                status_text.text(f"Processing... {progress}%")
                            
                            try:
                                # Run prediction with fixed batch size
                                status_text.text("Starting prediction...")
                                start_time = time.time()
                                
                                # Run the prediction
                                results_df = predictor.predict_from_file(
                                    input_path, 
                                    output_path, 
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
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="pred_download_results"
                                )
                                
                            except Exception as e:
                                st.error(f"Error during prediction: {str(e)}")
        
        with pred_tab2:
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
                {"Compartment": comp, "Description": descriptions.get(comp, f"Description for {comp}")}
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
        
        with pred_tab3:
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
            4. The Library & Mutation tab automatically formats sequences correctly
            """)

if __name__ == "__main__":
    main()