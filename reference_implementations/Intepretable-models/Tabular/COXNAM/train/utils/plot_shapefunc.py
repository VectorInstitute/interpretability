import numpy as np
import torch
import plotly.graph_objects as go

def plot_shape_functions_and_distributions(model, X, feature_names):
    """
    Creates an interactive Plotly figure with a dropdown menu to select a feature.
    For the selected feature, the plot shows:
      - The distribution (histogram) of the original (unnormalized) feature values.
      - The learned shape function (line) computed from the corresponding feature network.
    
    Parameters:
      model: Trained CoxNAM model.
      X: NumPy array of original feature values with shape (n_samples, n_features).
      feature_names: List of names for each feature.
    """
    num_features = X.shape[1]
    all_traces = []

    # Precompute traces for each feature
    for i in range(num_features):
        # Get original feature values for feature i
        feature_values = X[:, i]
        # Compute mean and standard deviation (assumed used for normalization during training)
        feature_mean = np.mean(feature_values)
        feature_std = np.std(feature_values)
        
        # --- Histogram Trace ---
        hist_trace = go.Histogram(
            x = feature_values,
            histnorm = 'probability density',  # Density normalization
            name = 'Distribution',
            opacity = 0.7,
            marker_color = 'blue'
        )
        
        # --- Shape Function Trace ---
        # Generate sample inputs in normalized space (the network expects normalized inputs)
        sample_inputs_norm = np.linspace(-3, 3, 100)
        sample_inputs_norm_tensor = torch.tensor(sample_inputs_norm.reshape(-1, 1), dtype=torch.float32)
        
        # Get the corresponding feature network and compute its output
        feature_network = model.feature_networks[i]
        feature_network.eval()
        with torch.no_grad():
            shape_values = feature_network(sample_inputs_norm_tensor).numpy().flatten()
        
        # Convert the normalized sample inputs back to the original scale.
        sample_inputs_orig = sample_inputs_norm * feature_std + feature_mean
        
        shape_trace = go.Scatter(
            x = sample_inputs_orig,
            y = shape_values,
            mode = 'lines',
            name = 'Shape Function',
            line = dict(color='red')
        )
        
        # Append both traces for this feature.
        all_traces.append(hist_trace)
        all_traces.append(shape_trace)

    # Create the figure with all traces.
    fig = go.Figure(data=all_traces)
    
    # Set initial visibility: only show traces for feature 0 (first two traces)
    total_traces = len(all_traces)  # equals 2 * num_features
    initial_visibility = [False] * total_traces
    initial_visibility[0] = True  # histogram for feature 0
    initial_visibility[1] = True  # shape function for feature 0

    # Set initial visibility of all traces.
    for j in range(total_traces):
        fig.data[j].visible = initial_visibility[j]

    # Create dropdown buttons – each button updates the visible traces.
    buttons = []
    for i in range(num_features):
        # Create a visibility list where only the two traces for feature i are visible.
        visibility = [False] * total_traces
        visibility[2 * i] = True      # histogram trace for feature i
        visibility[2 * i + 1] = True  # shape function trace for feature i
        
        button = dict(
            label = feature_names[i],
            method = 'update',
            args = [{'visible': visibility},
                    {'title': f'Feature: {feature_names[i]}'}]
        )
        buttons.append(button)
    
    # Add the dropdown menu to the layout.
    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            x=0.1,
            xanchor='left',
            y=1.15,
            yanchor='top'
        )],
        title = f'Feature: {feature_names[0]}',
        xaxis_title = 'Feature Value',
        yaxis_title = 'Density / Shape Function Output',
        font = dict(size=10),
        legend = dict(orientation='h', x=0.35, y=-0.15)
    )
    
    fig.show()

# Example usage:
# Assuming you have already loaded your data into a NumPy array X (of shape (n_samples, n_features)),
# a list of feature names in feature_names, and a trained model (with attribute feature_networks)
#
# plot_shape_functions_and_distributions_plotly(model, X, feature_names)