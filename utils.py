import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import base64
from pathlib import Path
from typing import List, Dict, Any

def generate_example_fasta():
    """
    Generate an example FASTA file with known condensate-localizing proteins.
    """
    example_data = """
>FUS|human|P35637|WT
MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSSYSSYGQSSYSGYSQSSYSGYSQSSYGSYGQSQNTGYGTQSTPQGYGSTGGYGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQSSSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNS
>TDP43|human|Q13148|WT
MSEYIRVTEDENDEPIEIPSEDDGTVLLSTVTAQFPGACGLRYRNPVSQCMRGVRLVEGILHAPDAGWGNLVYVVNYPKDNKRKMDETDASSAVKVKRAVQKTSDLIVLGLPWKTTEQDLKEYFSTFGEVLMVQVKKDLKTGHSKGFGFVRFTEYETQVKVMSQRHMIDGRWCDCKLPNSKQSQDEPLRSRKVFVGRCTEDMTEDELREFFSQYGDVMDVFIPKPFRAFAFVTFADDQIAQSLCGEDLIIKGISVHISNAEPKHNSNRQLERSGRFGGNPGGFGNQGGFGNSRGGGAGLGNNQGSNMGGGMNFGAFSINPAMMAAAQAALQSSWGMMGMLASQQNQSGPSGNNQNQGNMQREPNQAFGSGNNSYSGSNSGAAIGWGSASNAGSGSGFNGGFGSSMDSKSSGWGM
>hnRNPA1|human|P09651|WT
MSKSESPKEPEQLRKLFIGGLSFETTDESLRSHFEQWGTLTDCVVMRDPNTKRSRGFGFVTYATVEEVDAAMNARPHKVDGRVVEPKRAVSREDSQRPGAHLTVKKIFVGGIKEDTEEHHLRDYFEQYGKIEVIEIMTDRGSGKKRGFAFVTFDDHDSVDKIVIQKYHTVNGHNCEVRKALSKQEMASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGYGGSGDGYNGFGNDGGYGGGGPGYSGGSRGYGSGGQGYGNQGSGYGGSGSYDSYNNGGGGGFGGGSGSNFGGGGSYNDFGNYNNQSSNFGPMKGGNFGGRSSGPYGGGGQYFAKPRNQGGYGGSSSSSSYGSGRRF
"""
    
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    with open(data_dir / "example.fasta", "w") as f:
        f.write(example_data.strip())
    
    print(f"Example FASTA file created at: {data_dir / 'example.fasta'}")

def create_visualization(df, output_dir=None):
    """
    Create visualizations from prediction results.
    
    Args:
        df: DataFrame with prediction results
        output_dir: Directory to save visualizations (optional)
    
    Returns:
        Dictionary of figure objects
    """
    figures = {}
    
    # Extract score columns
    score_columns = [col for col in df.columns if col.endswith('_Score')]
    
    # 1. Heatmap of all predictions
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.3 + 2))
    im = ax.imshow(df[score_columns].values, cmap='viridis', aspect='auto')
    ax.set_xticks(np.arange(len(score_columns)))
    ax.set_yticks(np.arange(len(df)))
    ax.set_xticklabels([col.replace('_Score', '') for col in score_columns], rotation=45, ha="right")
    ax.set_yticklabels(df['Label'])
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Score", rotation=-90, va="bottom")
    plt.title("Condensate Localization Prediction Scores")
    plt.tight_layout()
    figures['heatmap'] = fig
    
    # 3. Individual protein profiles
    for i, row in df.iterrows():
        fig, ax = plt.subplots(figsize=(8, 5))
        scores = [row[col] for col in score_columns]
        compartments = [col.replace('_Score', '') for col in score_columns]
        
        # Sort by score descending
        sorted_idx = np.argsort(scores)[::-1]
        sorted_scores = [scores[i] for i in sorted_idx]
        sorted_compartments = [compartments[i] for i in sorted_idx]
        
        ax.bar(sorted_compartments, sorted_scores)
        ax.set_xticklabels(sorted_compartments, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(f"Prediction Profile: {row['Label']}")
        plt.tight_layout()
        figures[f'protein_{i}'] = fig
    
    # Save figures if output directory is provided
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        
        for name, fig in figures.items():
            fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches='tight')
    
    return figures

def dataframe_to_excel_download_link(df, filename="data.xlsx", text="Download Excel file"):
    """
    Generate a download link for a pandas DataFrame as an Excel file.
    
    Args:
        df: DataFrame to download
        filename: Name of the file to download
        text: Text for the download link
    
    Returns:
        HTML string with the download link
    """
    with tempfile.NamedTemporaryFile() as tmp:
        df.to_excel(tmp.name, index=False)
        tmp.seek(0)
        data = tmp.read()
    
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Generate example data when module is imported
if __name__ == "__main__":
    generate_example_fasta()