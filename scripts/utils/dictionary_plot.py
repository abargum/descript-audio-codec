import sys
import os
import time
import json
import shutil
import argparse
import torch
import random
import pickle
from sklearn.linear_model import LogisticRegression

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from rave.blocks3 import SpeakerRAVE
from rave.pqmf import CachedPQMF as PQMF
from utils import load_dict_from_txt, load_speaker_statedict

import librosa
import numpy as np
from sklearn import manifold
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  # For color conversion
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import yaml

def plot_interactive_pca(speaker_E, speaker_ID, speaker_AGE, speaker_GENDER, speaker_F0, speaker_LABEL):

    #Calculate PCA
    pca = PCA(n_components=2)
    components_pca = pca.fit_transform(speaker_E)

    # Calculate IQR for each PCA dimension to filter outliter
    q1 = np.percentile(components_pca, 25, axis=0)
    q3 = np.percentile(components_pca, 75, axis=0)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    valid_indices = np.all((components_pca >= lower_bound) & (components_pca <= upper_bound), axis=1)
        
    components_pca = components_pca[valid_indices]
    speaker_ID = [speaker_ID[i] for i in range(len(speaker_ID)) if valid_indices[i]]
    speaker_AGE = [speaker_AGE[i] for i in range(len(speaker_AGE)) if valid_indices[i]]
    speaker_GENDER = [speaker_GENDER[i] for i in range(len(speaker_GENDER)) if valid_indices[i]]
    speaker_F0 = [speaker_F0[i] for i in range(len(speaker_F0)) if valid_indices[i]]
    speaker_LABEL = [speaker_LABEL[i] for i in range(len(speaker_LABEL)) if valid_indices[i]]

    # Create decision boundary
    model = LogisticRegression()
    model.fit(components_pca, speaker_LABEL)
    w1, w2 = model.coef_[0]
    b = model.intercept_[0]

    LABELS = speaker_ID
    
    # Create a DataFrame for the PCA components and features
    df = pd.DataFrame({
        'Principal Component 1': components_pca[:, 0],
        'Principal Component 2': components_pca[:, 1],
        'Speaker': LABELS
    })

    df['Age'] = [speaker_AGE[i] for i in range(len(speaker_ID))]
    df['Gender'] = [speaker_GENDER[i] for i in range(len(speaker_ID))]
    df['Mean F0'] = [speaker_F0[i] for i in range(len(speaker_ID))]
    df['Label'] = [speaker_LABEL[i] for i in range(len(speaker_ID))]

    # Generate unique colors for each speaker
    unique_speakers = sorted(set(LABELS))
    num_speakers = len(unique_speakers)
    cmap = plt.cm.get_cmap("tab20", num_speakers)  # "tab20" colormap
    colors = [mcolors.rgb2hex(cmap(i)) for i in range(num_speakers)]  
    color_map = {speaker: colors[i] for i, speaker in enumerate(unique_speakers)} 

    # Find misclassified labels (embeddings on the wrong side of the boundary)
    df['Distance'] = (w1 * df['Principal Component 1'] + w2 * df['Principal Component 2'] + b) / np.sqrt(w1**2 + w2**2)
    df['Predicted'] = (df['Distance'] > 0).astype(int)  
    df['Misclassified'] = df['Label'] != df['Predicted']

    # Create the scatter plot with PCA data
    fig = px.scatter(
        df,
        x='Principal Component 1',
        y='Principal Component 2',
        color='Speaker',
        hover_data=['Speaker', 'Age', 'Gender', 'Mean F0'],
        title='Enhanced Interactive PCA Embedding',
        color_discrete_map=color_map  # Use the unique color mapping
    )
    
    # Add the decision boundary as a line
    x_vals = np.linspace(df['Principal Component 1'].min(), df['Principal Component 1'].max(), 500)
    y_vals = -(w1 / w2) * x_vals - b / w2
    
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        name='Decision Boundary',
        line=dict(color='black', dash='dash') 
    ))
    
    # Highlight misclassified points
    misclassified_df = df[df['Misclassified']]
    fig.add_trace(go.Scatter(
        x=misclassified_df['Principal Component 1'],
        y=misclassified_df['Principal Component 2'],
        mode='markers',
        name='Misclassified',
        marker=dict(color='red', size=10, symbol='x'),  
        hoverinfo='text',
        text=misclassified_df['Speaker']
    ))

    
    # Add text annotations for "female" and "male" side of boundary
    fig.add_annotation(
        x=df['Principal Component 1'].min(), 
        y=-(w1 / w2) * df['Principal Component 1'].min() - b / w2 + 1,  
        text="Female",  
        showarrow=False,
        font=dict(color="black", size=14)
    )
    
    fig.add_annotation(
        x=df['Principal Component 1'].max(),  # Position to the right of the line
        y=-(w1 / w2) * df['Principal Component 1'].max() - b / w2 - 1,  # Offset slightly for visibility
        text="Male",  # Label for male
        showarrow=False,
        font=dict(color="black", size=14)
    )

    # Save the plot to an HTML file
    fig.update_traces(marker=dict(size=9)) 
    fig.write_html('plots/interactive_avg_embeddings.html')
    print(f"Interactive plot saved to {'plots/interactive_avg_embeddings.html'}")
    

def main():
    random.seed(18)
    
    parser = argparse.ArgumentParser(description='inference')

    args = parser.parse_args()

    file = 'VCTK-Corpus/speaker-info.txt'
    info_dict = load_dict_from_txt(file)

    pqmf = PQMF(attenuation = 100, n_band = 16)

    speaker_encoder = SpeakerRAVE()
    spk_state, pqmf_state = load_speaker_statedict("scripts/rave/model000000075.model")
    speaker_encoder.load_state_dict(spk_state)
    speaker_encoder.eval()

    file_path = 'scripts/utils/speaker_emb_dict.pkl'
    with open(file_path, 'rb') as file:
        speaker_dict = pickle.load(file)

    speaker_E = []
    speaker_ID = []
    speaker_AGE = []
    speaker_GENDER = []
    speaker_LABEL = []
    speaker_F0 = []

    for name, item in speaker_dict.items():
        if name != 'p280':
            speaker_ID.append(name) 
            speaker_F0.append(item['f0_mean'])
            speaker_E.append(item['avg_emb'])
            speaker_AGE.append(info_dict[str(name)]['AGE'])
            speaker_GENDER.append(info_dict[str(name)]['GENDER'])
            if info_dict[str(name)]['GENDER'] == 'F':
                speaker_LABEL.append(0)
            else:
                speaker_LABEL.append(1)

    speaker_E = np.array(speaker_E).reshape(-1, 256)
    
    plot_interactive_pca(speaker_E, speaker_ID, speaker_AGE, speaker_GENDER, speaker_F0, speaker_LABEL)

if __name__ == "__main__":
    main()