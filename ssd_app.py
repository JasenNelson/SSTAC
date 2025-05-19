import streamlit as st 
import pandas as pd
import numpy as np
import scipy.stats
import plotly.graph_objects as go
from io import StringIO

# --- Configuration --- (Keep this section as is)
ECOTOX_EXPECTED_COLS = {
    'cas': 'test_cas',            # CAS Registry Number
    'chemical': 'chemical_name',  # Chemical Name
    'species_sci': 'species_scientific_name', # Species Scientific Name
    'species_common': 'species_common_name', # Species Common Name (Optional)
    'group': 'species_group',     # Taxonomic Group
    'endpoint': 'endpoint',       # Endpoint code
    'effect': 'effect',           # Effect measured
    'conc_mean': 'conc1_mean',    # Mean concentration value
    'conc_unit': 'conc1_unit',    # Concentration unit
}
TAXONOMIC_MAPPING = {
    'Fish': ['Fish'],
    'Invertebrate': ['Aquatic Invertebrates', 'Crustaceans', 'Insects', 'Molluscs', 'Worms', 'Zooplankton'],
    'Algae/Plant': ['Algae', 'Aquatic Plants', 'Plants (Seedlings)', 'Plants']
}
ACUTE_ENDPOINTS = ['LC50', 'EC50']
CHRONIC_ENDPOINTS = ['NOEC', 'LOEC', 'EC10']

# --- Helper Functions --- (Keep these as is: map_taxonomic_group, get_distribution, calculate_ssd, create_ssd_plot)
def map_taxonomic_group(ecotox_group):
    """Maps detailed ECOTOX group to broader category."""
    for broad_group, detailed_list in TAXONOMIC_MAPPING.items():
        if ecotox_group in detailed_list:
            return broad_group
    return "Other"

def get_distribution(dist_name):
    """Returns the scipy.stats distribution object based on name."""
    if dist_name.lower() == 'log-normal':
        return scipy.stats.norm
    elif dist_name.lower() == 'log-logistic':
        return scipy.stats.logistic
    else:
        raise ValueError(f"Unsupported distribution: {dist_name}")

def calculate_ssd(data, species_col, value_col, dist_name, p_value):
    """ Calculates SSD parameters and HCp. """
    if data.empty or data[value_col].isnull().all():
        return None, None, None, "No valid data points for SSD."
    valid_data = data[data[value_col] > 0].copy()
    if valid_data.empty:
        return None, None, None, "No positive toxicity values found for log transformation."
    valid_data['log10_value'] = np.log10(valid_data[value_col])
    log_values = valid_data['log10_value'].dropna()
    if len(log_values) < 3:
         return None, None, None, f"Insufficient data points ({len(log_values)}) for distribution fitting."
    dist = get_distribution(dist_name)
    try:
        params = dist.fit(log_values)
    except Exception as e:
        return None, None, None, f"Error fitting distribution '{dist_name}': {e}"
    log_hcp = dist.ppf(p_value / 100.0, *params)
    hcp = 10**log_hcp
    n = len(log_values)
    sorted_log_values = np.sort(log_values)
    empirical_cdf = (np.arange(1, n + 1) / (n + 1)) * 100
    prob_range = np.linspace(0.001, 0.999, 200)
    fitted_log_values = dist.ppf(prob_range, *params)
    fitted_cdf_percent = prob_range * 100
    plot_data = {
        'empirical_log_values': sorted_log_values,
        'empirical_cdf_percent': empirical_cdf,
        'fitted_log_values': fitted_log_values,
        'fitted_cdf_percent': fitted_cdf_percent,
        'log_hcp': log_hcp,
        'hcp_p_percent': p_value,
        'species': valid_data[species_col]
    }
    return hcp, params, plot_data, None

def create_ssd_plot(plot_data, hcp, p_value, dist_name, unit):
    """ Generates the SSD Plotly figure. """
    if plot_data is None: return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_data['empirical_log_values'], y=plot_data['empirical_cdf_percent'], mode='markers', name='Species Data',
        marker=dict(color='blue', size=8),
        hovertext=[f"Species: {sp}<br>Log10 Conc: {log_val:.2f}" for sp, log_val in zip(plot_data['species'], plot_data['empirical_log_values'])], hoverinfo='text'
    ))
    fig.add_trace(go.Scatter(
        x=plot_data['fitted_log_values'], y=plot_data['fitted_cdf_percent'], mode='lines', name=f'Fitted {dist_name} CDF', line=dict(color='red', dash='dash')
    ))
    fig.add_hline(y=p_value, line=dict(color='grey', dash='dot'), name=f'{p_value}% Level')
    fig.add_vline(x=plot_data['log_hcp'], line=dict(color='grey', dash='dot'), name=f'HC{p_value}')
    fig.add_trace(go.Scatter(x=[plot_data['log_hcp']], y=[p_value], mode='markers', marker=dict(color='red', size=10, symbol='x'), name=f'HC{p_value}'))
    fig.update_layout(
        title='Species Sensitivity Distribution (SSD)', xaxis_title=f'Concentration (Log10 {unit})', yaxis_title='Percent of Species Affected (%)',
        legend_title='Legend', xaxis=dict(tickformat=".2f"), yaxis=dict(range=[0, 100]), hovermode='closest'
    )
    return fig

# --- Helper Function to Read File Header / Chemical Names ---
@st.cache_data(show_spinner=False) # Cache results to avoid re-reading on every interaction
def get_chemical_options(uploaded_file):
    """Reads the file to extract unique chemical names for the dropdown."""
    if uploaded_file is None:
        return ["-- Upload File First --"]

    try:
        file_content = uploaded_file.getvalue()
        df_chem = None
        chem_col = ECOTOX_EXPECTED_COLS['chemical']
        # Try reading as CSV first (more common for processed files)
        try:
            df_chem = pd.read_csv(StringIO(file_content.decode('latin-1')), sep=',', usecols=[chem_col]) # Read only chemical column
        except (ValueError, KeyError): # Handle cases where column doesn't exist or file is not comma-separated
             pass # Try next format

        # Fallback to tab-separated
        if df_chem is None:
            try:
                 uploaded_file.seek(0) # Reset file pointer
                 df_chem = pd.read_csv(StringIO(file_content.decode('latin-1')), sep='\t', usecols=[chem_col])
            except (ValueError, KeyError):
                 return ["-- Error: Cannot find 'chemical_name' column or parse file --"]
            except Exception: # Catch other potential parsing errors
                 return ["-- Error: Cannot parse file header --"]

        if df_chem is not None and chem_col in df_chem.columns:
            unique_names = df_chem[chem_col].dropna().astype(str).str.strip().unique()
            if len(unique_names) > 0:
                sorted_names = sorted([name for name in unique_names if name]) # Exclude empty strings
                return ["-- Select Chemical --"] + sorted_names
            else:
                return ["-- No Chemical Names Found --"]
        else:
             return ["-- Error: 'chemical_name' column missing --"]

    except Exception as e:
        # General error catching during the read attempt
        st.error(f"Error reading file for chemical list: {e}")
        return ["-- Error Reading File --"]

# --- Streamlit App UI ---

st.set_page_config(layout="wide")
st.title("🧪 Species Sensitivity Distribution (SSD) Generator")
st.markdown("""
Upload your **processed** ecotoxicity data file (e.g., a `.csv` containing the required columns:
`test_cas`, `chemical_name`, `species_scientific_name`, `species_common_name`, `species_group`,
`endpoint`, `effect`, `conc1_mean`, `conc1_unit`). Select the chemical from the dropdown and
configure parameters to generate the SSD.
""")

# --- Initialize Session State ---
# Use session state to prevent resetting dropdown when other widgets change
if 'selected_chemical' not in st.session_state:
    st.session_state.selected_chemical = "-- Select Chemical --"
if 'file_processed_chem_list' not in st.session_state:
     st.session_state.file_processed_chem_list = None

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("⚙️ Settings")

    # Store uploaded file info in session state to manage dropdown refresh
    uploaded_file = st.file_uploader("1. Upload Processed Data File", type=['csv', 'txt']) # Prioritize CSV

    chemical_options = ["-- Upload File First --"]
    if uploaded_file is not None:
         # Check if file has changed or if list hasn't been generated yet
         if uploaded_file != st.session_state.get('last_uploaded_file', None) or st.session_state.file_processed_chem_list is None:
              st.session_state.last_uploaded_file = uploaded_file
              with st.spinner("Reading chemical list..."):
                    st.session_state.file_processed_chem_list = get_chemical_options(uploaded_file)
              # Reset selection if file changes
              st.session_state.selected_chemical = st.session_state.file_processed_chem_list[0] if st.session_state.file_processed_chem_list else "-- Error --"

         chemical_options = st.session_state.file_processed_chem_list or ["-- Error Reading File --"]


    # *** MODIFIED: Use Selectbox ***
    # Find the index of the currently selected chemical to maintain state
    current_selection_index = 0
    if st.session_state.selected_chemical in chemical_options:
         current_selection_index = chemical_options.index(st.session_state.selected_chemical)
    elif len(chemical_options) > 0:
         # Fallback if previous selection is somehow not in the new list
         current_selection_index = 0
         st.session_state.selected_chemical = chemical_options[0]


    selected_chemical = st.selectbox(
        "2. Select Chemical Name",
        options=chemical_options,
        index=current_selection_index, # Use index to set default/current value
        key='selected_chemical', # Use key to link to session state
        disabled=(uploaded_file is None), # Disable if no file or error
        help="Select the chemical from the list derived from your uploaded file."
    )

    st.subheader("SSD Parameters")

    # --- Keep the rest of the sidebar inputs the same ---
    endpoint_type = st.radio(
        "3. Endpoint Type", ('Acute (LC50, EC50)', 'Chronic (NOEC, LOEC, EC10)'), index=0,
        help="Select the general type of endpoint to include."
    )
    min_species = st.number_input(
        "4. Minimum Number of Species", min_value=3, value=5, step=1,
        help="Minimum unique species required after filtering."
    )
    required_taxa_broad = st.multiselect(
        "5. Required Taxonomic Groups", options=list(TAXONOMIC_MAPPING.keys()), default=list(TAXONOMIC_MAPPING.keys())[:3],
        help="Select the broad taxonomic groups that *must* be represented."
    )
    data_handling = st.radio(
        "6. Handle Multiple Values per Species", ('Use Geometric Mean', 'Use Most Sensitive (Minimum Value)'), index=0,
        help="How to aggregate multiple data points for the same species."
    )
    distribution_fit = st.selectbox(
        "7. Distribution for Fitting", ('Log-Normal', 'Log-Logistic'), index=0,
        help="Statistical distribution to fit to the log-transformed data."
    )
    hcp_percentile = st.number_input(
        "8. Hazard Concentration (HCp) Percentile", min_value=0.1, max_value=99.9, value=5.0, step=0.1, format="%.1f",
        help="The percentile 'p' for which to calculate the HCp (e.g., 5 for HC5)."
    )

    # *** MODIFIED: Button enabling logic ***
    is_ready_to_generate = (
        uploaded_file is not None and
        selected_chemical is not None and
        not selected_chemical.startswith("--") # Ensure a valid chemical is selected
    )
    generate_button = st.button("🚀 Generate SSD", disabled=(not is_ready_to_generate))

# --- Main Area for Processing and Output ---
results_area = st.container()
plot_area = st.container()

if generate_button and is_ready_to_generate: # Check readiness flag
    try:
        # Read the uploaded file AGAIN for full processing
        # Reset file pointer just in case
        st.session_state.last_uploaded_file.seek(0)
        file_content = st.session_state.last_uploaded_file.getvalue()
        df = None
        # Try CSV first
        try:
            df = pd.read_csv(StringIO(file_content.decode('latin-1')), sep=',') # Assume comma for CSV
            # Quick check if columns expected exist after comma read
            expected_cols_present = all(col in df.columns for col in ECOTOX_EXPECTED_COLS.values())
            if not expected_cols_present:
                 st.write("Info: Comma-separated read didn't find all expected columns, trying tab...")
                 df = None # Reset df to trigger tab read
        except Exception as e_csv:
             st.write(f"Info: Failed reading as comma-separated ({e_csv}), trying tab-separated...")
             df = None # Ensure df is None if CSV read fails

        # Fallback to Tab
        if df is None:
            try:
                 st.session_state.last_uploaded_file.seek(0) # Reset again
                 df = pd.read_csv(StringIO(file_content.decode('latin-1')), sep='\t') # Try tab
            except Exception as e_tab:
                 st.error(f"❌ Error parsing file: Could not read as CSV or TSV. Check format and separator. Error: {e_tab}")
                 st.stop()

        # Check if essential columns exist
        missing_cols = [col_name for col_key, col_name in ECOTOX_EXPECTED_COLS.items() if col_name not in df.columns]
        if missing_cols:
            st.error(f"❌ Error: The uploaded file is missing required columns: {', '.join(missing_cols)}. "
                     f"Expected columns based on configuration: {', '.join(ECOTOX_EXPECTED_COLS.values())}")
            st.stop()

        st.write("---")
        st.subheader("Processing Steps:")

        # --- 1. Initial Filtering (Chemical & Endpoint Type) ---
        # *** MODIFIED: Filter by EXACT chemical name from dropdown ***
        name_col = ECOTOX_EXPECTED_COLS['chemical']
        st.write(f"1. Filtering for Chemical: '{selected_chemical}'")
        # Ensure column is string type and strip whitespace just in case
        df[name_col] = df[name_col].astype(str).str.strip()
        # Exact match filtering
        chem_filter = (df[name_col] == selected_chemical)
        filtered_df = df[chem_filter].copy()
        # --- End Modification ---

        if filtered_df.empty:
            st.warning(f"⚠️ No data found for chemical '{selected_chemical}'.")
            st.stop()

        st.write(f"   Found {len(filtered_df)} initial records for the chemical.")

        # --- Steps 2-6: Remain largely the same ---
        # (Endpoint filtering, Data Cleaning, Aggregation, Requirement Checks, SSD Calculation)
        # ... (Keep the existing logic for these steps) ...
        # Make sure to use 'filtered_df' as the starting point

        # 2. Endpoint Filter
        endpoint_col = ECOTOX_EXPECTED_COLS['endpoint']
        if endpoint_type == 'Acute (LC50, EC50)': selected_endpoints = ACUTE_ENDPOINTS
        else: selected_endpoints = CHRONIC_ENDPOINTS
        st.write(f"2. Filtering for {endpoint_type.split(' ')[0]} Endpoints: {', '.join(selected_endpoints)}")
        filtered_df = filtered_df[filtered_df[endpoint_col].isin(selected_endpoints)]
        if filtered_df.empty: st.warning(f"⚠️ No data found for selected endpoints."); st.stop()
        st.write(f"   {len(filtered_df)} records remaining.")

        # 3. Data Cleaning / Prep
        st.write("3. Cleaning and Preparing Data:")
        value_col = ECOTOX_EXPECTED_COLS['conc_mean']
        unit_col = ECOTOX_EXPECTED_COLS['conc_unit']
        species_col = ECOTOX_EXPECTED_COLS['species_sci']
        group_col = ECOTOX_EXPECTED_COLS['group']
        filtered_df[value_col] = pd.to_numeric(filtered_df[value_col], errors='coerce')
        initial_rows = len(filtered_df)
        filtered_df.dropna(subset=[species_col, value_col], inplace=True)
        dropped_rows = initial_rows - len(filtered_df)
        if dropped_rows > 0: st.write(f"   Dropped {dropped_rows} rows with missing species or concentration.")
        valid_units = filtered_df[unit_col].dropna().unique()
        if len(valid_units) > 1: st.warning(f"⚠️ Warning: Multiple units found: {', '.join(valid_units)}. Using most frequent for labeling.")
        data_unit = filtered_df[unit_col].mode()[0] if not filtered_df[unit_col].mode().empty else "units"
        st.write(f"   Using unit for results: '{data_unit}'")
        if filtered_df.empty: st.warning("⚠️ No valid numeric data after cleaning."); st.stop()

        # 4. Aggregate per Species
        st.write(f"4. Aggregating data per species using: '{data_handling}'")
        grouped = filtered_df.groupby(species_col)
        aggregated_data_list = []
        for name, group in grouped:
            species_group_val = group[group_col].iloc[0] if group_col in group.columns else "Unknown"
            if data_handling == 'Use Geometric Mean':
                positive_values = group[value_col][group[value_col] > 0]
                agg_value = scipy.stats.gmean(positive_values) if not positive_values.empty else np.nan
            else: # Most Sensitive
                agg_value = group[value_col].min()
            if pd.notna(agg_value):
                 aggregated_data_list.append({
                     species_col: name, 'broad_group': map_taxonomic_group(species_group_val), 'aggregated_value': agg_value
                 })
        species_df = pd.DataFrame(aggregated_data_list)
        species_df.dropna(subset=['aggregated_value'], inplace=True)
        st.write(f"   Data aggregated down to {len(species_df)} unique species.")

        # 5. Check Data Requirements
        st.write("5. Checking data requirements:")
        final_species_count = len(species_df)
        st.write(f"   - Found {final_species_count} species.")
        if final_species_count < min_species: st.error(f"❌ Error: Insufficient species ({final_species_count}). Minimum required: {min_species}."); st.stop()
        else: st.success(f"   ✓ Species count requirement met (>= {min_species}).")
        present_taxa = species_df['broad_group'].unique()
        missing_required_taxa = [taxon for taxon in required_taxa_broad if taxon not in present_taxa]
        st.write(f"   - Present broad taxonomic groups: {', '.join(present_taxa)}")
        if not required_taxa_broad: st.write("   - No specific taxonomic groups required.")
        elif not missing_required_taxa: st.success(f"   ✓ All required taxonomic groups present.")
        else: st.error(f"❌ Error: Missing required taxonomic groups: {', '.join(missing_required_taxa)}."); st.stop()

        # 6. Perform SSD Calculation
        st.write(f"6. Fitting SSD using '{distribution_fit}' and calculating HC{hcp_percentile}:")
        hcp_value, fit_params, plot_data_dict, error_msg = calculate_ssd(
            data=species_df, species_col=species_col, value_col='aggregated_value',
            dist_name=distribution_fit, p_value=hcp_percentile
        )
        if error_msg: st.error(f"❌ Error during SSD calculation: {error_msg}"); st.stop()
        st.success("   ✓ SSD calculation successful.")

        # --- Display Results --- (Keep as is)
        with results_area:
            st.subheader("📊 Results")
            st.metric(label=f"Hazard Concentration HC{hcp_percentile}", value=f"{hcp_value:.4g} {data_unit}")
        with plot_area:
            st.subheader("📈 SSD Plot")
            ssd_fig = create_ssd_plot(plot_data_dict, hcp_value, hcp_percentile, distribution_fit, data_unit)
            st.plotly_chart(ssd_fig, use_container_width=True)
        st.subheader("Species Data Used for SSD")
        st.dataframe(species_df[[species_col, 'broad_group', 'aggregated_value']].rename(
            columns={'aggregated_value': f'Toxicity Value ({data_unit})', 'broad_group': 'Taxonomic Group'}
        ).round(4))

    # --- Keep Error Handling --- (Keep as is)
    except pd.errors.ParserError as e: st.error(f"❌ File Parsing Error: {e}")
    except KeyError as e: st.error(f"❌ Column Not Found Error: '{e}'. Check file header.")
    except ValueError as e: st.error(f"❌ Value Error: {e}")
    except Exception as e: st.error(f"❌ An unexpected error occurred: {e}"); st.exception(e)

elif generate_button and not is_ready_to_generate:
    st.warning("⚠️ Please ensure a file is uploaded and a chemical is selected from the dropdown.")

else:
    st.info("Upload a file using the sidebar to populate the chemical list and enable analysis.")