import os
import pickle
import torch
import pandas as pd
from pathlib import Path
from argparse import Namespace
import sys
from typing import List, Tuple, Dict, Any, Optional

# Get the current directory of this script
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Add the protgps directory to the system path
sys.path.append(str(SCRIPT_DIR))

# Import after path setup
from protgps.utils.loading import get_object

# Constants
COMPARTMENT_CLASSES = [
    "nuclear_speckle", "p-body", "pml-bdoy", "post_synaptic_density",
    "stress_granule", "chromosome", "nucleolus", "nuclear_pore_complex",
    "cajal_body", "rna_granule", "cell_junction", "transcriptional"
]

class ProtGPSPredictor:
    """
    Class for making predictions with the ProtGPS model.
    """
    def __init__(self, device=None):
        """
        Initialize the predictor.
        
        Args:
            device: The device to use for prediction (cuda or cpu).
        """
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        print(f"Using device: {self.device}")
        
    def load_model(self) -> None:
        """
        Load the ProtGPS model from the models directory.
        """
        # Set paths relative to script directory
        args_file = SCRIPT_DIR / "models" / "protgps" / "32bf44b16a4e770a674896b81dfb3729.args"
        checkpoint_file = SCRIPT_DIR / "models" / "protgps" / "32bf44b16a4e770a674896b81dfb3729epoch=26.ckpt"
        esm_dir = SCRIPT_DIR / "models" / "esm2"
        
        # Load args
        args = Namespace(**pickle.load(open(args_file, 'rb')))
        args.model_path = str(checkpoint_file)
        args.pretrained_hub_dir = str(esm_dir)
        
        # Load model
        model = get_object(args.lightning_name, "lightning")(args)
        model = model.load_from_checkpoint(
            checkpoint_path=args.model_path,
            strict=not args.relax_checkpoint_matching,
            **{"args": args},
        )
        model.eval()
        model = model.to(self.device)
        
        self.model = model
        print("Model loaded successfully!")
        
    def parse_fasta(self, fasta_path: str) -> Tuple[List[str], List[str]]:
        """
        Parse sequences and labels from a FASTA file.
        
        Args:
            fasta_path: Path to the FASTA file.
            
        Returns:
            Tuple of (sequences, labels).
        """
        sequences = []
        labels = []
        
        with open(fasta_path, 'r') as f:
            current_label = None
            current_sequence = ""
            
            for line in f:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                    
                # Header line
                if line.startswith('>'):
                    # Save previous sequence if any
                    if current_label and current_sequence:
                        sequences.append(current_sequence)
                        labels.append(current_label)
                    
                    # Extract components from the header
                    # Format: >gene|organism|uniprot_id|mutation_status
                    header_parts = line[1:].split('|')
                    if len(header_parts) >= 4:
                        gene, organism, uniprot_id, mutation_status = header_parts[:4]
                        # Format label to match Streamlit app's format
                        current_label = f"{organism}, {gene}, {mutation_status}"
                    else:
                        # Fallback if header doesn't match expected format
                        current_label = line[1:]
                    
                    current_sequence = ""
                else:
                    # Sequence line - accumulate sequence without spaces
                    current_sequence += line
            
            # Add the last sequence if any
            if current_label and current_sequence:
                sequences.append(current_sequence)
                labels.append(current_label)
        
        return sequences, labels
    
    @torch.no_grad()
    def predict(self, sequences: List[str], batch_size: int = 4, 
               progress_callback=None) -> torch.Tensor:
        """
        Run predictions on the sequences.
        
        Args:
            sequences: List of protein sequences.
            batch_size: Batch size for prediction.
            progress_callback: Function to call with progress updates (0-100).
            
        Returns:
            Tensor of prediction scores.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        scores = []
        total_batches = len(sequences) // batch_size + (1 if len(sequences) % batch_size > 0 else 0)
        
        for i in range(0, len(sequences), batch_size):
            # Calculate progress
            progress = int(100 * i / len(sequences))
            if progress_callback:
                progress_callback(progress)
                
            batch = sequences[i:(i + batch_size)]
            out = self.model.model({"x": batch})    
            s = torch.sigmoid(out['logit']).to("cpu")
            scores.append(s)
            
        # Final progress update
        if progress_callback:
            progress_callback(100)
            
        scores = torch.vstack(scores)
        return torch.round(scores, decimals=3)
    
    def predict_from_file(self, fasta_path: str, output_path: str, batch_size: int = 4,
                         progress_callback=None) -> pd.DataFrame:
        """
        Run predictions on sequences in a FASTA file and save results to Excel.
        
        Args:
            fasta_path: Path to the FASTA file.
            output_path: Path to save the Excel results.
            batch_size: Batch size for prediction.
            progress_callback: Function to call with progress updates.
            
        Returns:
            DataFrame with the results.
        """
        # Parse FASTA file
        sequences, labels = self.parse_fasta(fasta_path)
        print(f"Found {len(sequences)} sequences to process")
        
        # Run predictions
        scores = self.predict(sequences, batch_size, progress_callback)
        
        # Format results
        data = {"Label": labels, "Sequence": sequences}
        for j, compartment in enumerate(COMPARTMENT_CLASSES):
            data[f"{compartment.upper()}_Score"] = scores[:, j].tolist()
        
        # Create DataFrame and save to Excel
        df = pd.DataFrame(data)
        df.to_excel(output_path, index=False)
        
        return df
        
    def get_compartment_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions for each compartment class.
        
        Returns:
            Dictionary mapping compartment names to descriptions.
        """
        return {
            "nuclear_speckle": "Nuclear speckles are subnuclear structures enriched in pre-mRNA splicing factors",
            "p-body": "Processing bodies involved in mRNA decay",
            "pml-bdoy": "PML bodies are nuclear bodies implicated in a wide range of processes",
            "post_synaptic_density": "Specialized structure in neuronal synapses",
            "stress_granule": "Cytoplasmic foci formed upon cellular stress",
            "chromosome": "Organized structure of DNA and proteins",
            "nucleolus": "Nuclear compartment where ribosome biogenesis occurs",
            "nuclear_pore_complex": "Complex that mediates transport between nucleus and cytoplasm",
            "cajal_body": "Nuclear organelles involved in snRNP biogenesis",
            "rna_granule": "Various RNA-protein assemblies",
            "cell_junction": "Specialized structure forming connection between cells",
            "transcriptional": "Involved in transcriptional regulation"
        }