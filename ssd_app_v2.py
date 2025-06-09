# File updated to trigger commit (no functional change)
import streamlit as st 
import re
import logging
import os
import sys
import warnings
from functools import wraps

# Completely disable all warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# Set environment variables to suppress Supabase messages
os.environ['SUPABASE_DEBUG'] = '0'
os.environ['SUPABASE_VERBOSE_ERRORS'] = '0'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Configure logging to suppress all logs
logging.basicConfig(
    level=logging.CRITICAL,
    format='',
    handlers=[logging.NullHandler()]
)
logging.disable(logging.CRITICAL)

# Suppress all loggers
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).handlers = []
    logging.getLogger(logger_name).addHandler(logging.NullHandler())
    logging.getLogger(logger_name).propagate = False
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

# Redirect stdout and stderr to /dev/null
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

# Function to temporarily restore stdout/stderr for Streamlit
def restore_output():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

# Restore output for Streamlit
restore_output()

# Save original print
_original_print = print
_original_stdout = sys.stdout
_original_stderr = sys.stderr

class SuppressOutput:
    def __enter__(self):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if sys.stdout != sys.__stdout__:  # Only close if it's our file
            sys.stdout.close()
        if sys.stderr != sys.__stderr__:
            sys.stderr.close()
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        return False

def suppress_output(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with SuppressOutput():
            return func(*args, **kwargs)
    return wrapper

# Apply suppression to print
print = suppress_output(print)

# Configure logging to suppress unwanted messages
logging.basicConfig(
    level=logging.ERROR,  # Only show ERROR and above
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.NullHandler()]  # Don't output any logs
)

# Suppress all loggers
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).addHandler(logging.NullHandler())
    logging.getLogger(logger_name).propagate = False
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

# Disable debug logging for supabase
os.environ['SUPABASE_DEBUG'] = 'false'
os.environ['SUPABASE_VERBOSE_ERRORS'] = 'false'
os.environ['PYTHONUNBUFFERED'] = '1'

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# --- Table Name Constants ---
TABLE_CHEMICALS = "toxicology_data"

def validate_input(text, max_length=100, allowed_chars=r'[^a-zA-Z0-9\s\-_,.]'):
    """
    Validate and sanitize user input to prevent injection attacks.
    Returns sanitized text or raises ValueError if invalid.
    """
    if not text or not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text) > max_length:
        raise ValueError(f"Input too long. Maximum {max_length} characters allowed.")
    if re.search(allowed_chars, text):
        raise ValueError("Invalid characters in input. Only letters, numbers, spaces, hyphens, underscores, periods, and commas are allowed.")
    return text

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm, logistic, weibull_min
from io import StringIO
from typing import Tuple, Dict
import supabase
from st_supabase_connection import SupabaseConnection
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set page configuration first
st.set_page_config(layout="wide")


@st.cache_resource(show_spinner=False)
def initialize_supabase_connection():
    """Initialize and test Supabase connection with proper error handling."""
    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        with open(os.devnull, 'w') as f:
            sys.stdout, sys.stderr = f, f
            from supabase import create_client as _create_client
            supabase_url = st.secrets.get("connections", {}).get("supabase", {}).get("url") or os.environ.get("SUPABASE_URL")
            supabase_key = st.secrets.get("connections", {}).get("supabase", {}).get("key") or os.environ.get("SUPABASE_KEY")
            if not supabase_url or not supabase_key: return None
            client = _create_client(supabase_url, supabase_key)
            client.table("toxicology_data").select("id").limit(1).execute()
            return client
    except Exception:
        return None
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

supabase_conn = initialize_supabase_connection()

# --- App Configuration ---
TAXONOMIC_MAPPING = {
    'Fish': ['Fish'],
    'Invertebrate': ['Aquatic Invertebrates', 'Crustaceans', 'Insects', 'Molluscs', 'Worms', 'Zooplankton'],
    'Plant': ['Algae', 'Aquatic Plants', 'Plants (Seedlings)', 'Plants'],
    'Amphibian': ['Amphibians']
}

ENDPOINT_PREFERENCE_LONG_TERM = {
    'EC10': 1, 'IC10': 1, 'EC11-25': 2, 'IC11-25': 2, 'MATC': 3, 'NOEC': 4,
    'LOEC': 5, 'EC26-49': 6, 'IC26-49': 6, 'EC50': 7, 'IC50': 7, 'LC50': 8
}

# --- Helper Functions ---
@st.cache_data
def get_chemical_options_from_file(uploaded_file):
    """Reads an uploaded file and returns a list of unique chemical names."""
    if not uploaded_file: return []
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        if 'chemical_name' in df.columns:
            return sorted(df[df['chemical_name'].notna()]['chemical_name'].unique())
    except Exception:
        return []
    return []

@st.cache_data(ttl=600)
def search_chemicals_in_db(_supabase_client, search_term):
    """Searches for chemical names in the database."""
    if not search_term:
        return []
    try:
        response = _supabase_client.table("toxicology_data").select("chemical_name").ilike("chemical_name", f"%{search_term}%").execute()
        if response.data:
            unique_names = sorted(list(set(item['chemical_name'] for item in response.data)))
            return unique_names
    except Exception as e:
        st.error(f"Database search failed: {e}")
        return []
    return []


def map_taxonomic_group(ecotox_group):
    """Maps detailed ECOTOX group to broader CCME category."""
    for broad_group, detailed_list in TAXONOMIC_MAPPING.items():
        if ecotox_group in detailed_list:
            return broad_group
    return "Other"

def get_endpoint_rank(endpoint: str) -> int:
    """Returns the preference rank for a given endpoint string."""
    endpoint_upper = endpoint.upper()
    if 'EC' in endpoint_upper or 'IC' in endpoint_upper:
        try:
            num_part = re.search(r'\d+', endpoint_upper)
            if num_part:
                val = int(num_part.group(0))
                if 11 <= val <= 25: return ENDPOINT_PREFERENCE_LONG_TERM.get('EC11-25', 99)
                if 26 <= val <= 49: return ENDPOINT_PREFERENCE_LONG_TERM.get('EC26-49', 99)
        except (ValueError, TypeError): pass
    return ENDPOINT_PREFERENCE_LONG_TERM.get(endpoint_upper, 99)

def calculate_ssd(data, species_col, value_col, p_value, dist_name='Log-Normal'):
    """Calculates SSD parameters and HCp for various distributions."""
    valid_data = data[data[value_col] > 0].copy()
    if valid_data.empty:
        return None, None, None, "No positive concentrations for SSD calculation."
    
    params, hcp, fitted_y_cdf, x_range = None, None, None, None
    log_data = np.log(valid_data[value_col])
    
    try:
        if dist_name == 'Log-Normal':
            mean, std = norm.fit(log_data)
            params = (mean, std)
            hcp = np.exp(norm.ppf(p_value, loc=mean, scale=std))
            x_range = np.linspace(log_data.min() - 1, log_data.max() + 1, 200)
            fitted_y_cdf = norm.cdf(x_range, loc=mean, scale=std)

        elif dist_name == 'Log-Logistic':
            loc, scale = logistic.fit(log_data)
            params = (loc, scale)
            hcp = np.exp(logistic.ppf(p_value, loc=loc, scale=scale))
            x_range = np.linspace(log_data.min() - 1, log_data.max() + 1, 200)
            fitted_y_cdf = logistic.cdf(x_range, loc=loc, scale=scale)

        elif dist_name == 'Weibull':
            shape, loc, scale = weibull_min.fit(valid_data[value_col], floc=0)
            params = (shape, loc, scale)
            hcp = weibull_min.ppf(p_value, shape, loc=loc, scale=scale)
            x_range = np.linspace(valid_data[value_col].min() * 0.1, valid_data[value_col].max() * 1.1, 200)
            fitted_y_cdf = weibull_min.cdf(x_range, shape, loc=loc, scale=scale)
        
        plot_data = {
            'empirical_values': np.sort(valid_data[value_col]),
            'empirical_cdf_percent': (np.arange(1, len(valid_data) + 1) / (len(valid_data) + 1)) * 100,
            'fitted_x_range': x_range,
            'fitted_y_cdf_percent': fitted_y_cdf * 100,
            'species': valid_data[species_col].tolist(),
            'p_value': p_value,
            'dist_name': dist_name
        }
        return hcp, params, plot_data, None
    except Exception as e:
        return None, None, None, f"Error during {dist_name} fit: {str(e)}"

def create_ssd_plot(plot_data, hcp, unit):
    """Generates the SSD Plotly figure."""
    if plot_data is None: return go.Figure()
    
    dist_name = plot_data['dist_name']
    is_log_scale = dist_name in ['Log-Normal', 'Log-Logistic']
    
    empirical_x = plot_data['empirical_values']
    fitted_x = np.exp(plot_data['fitted_x_range']) if is_log_scale else plot_data['fitted_x_range']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=empirical_x, y=plot_data['empirical_cdf_percent'], mode='markers', name='Species Data',
        marker=dict(color='#2196F3', size=8, symbol='circle'),
        hovertext=[f"Species: {sp}<br>Conc: {x:.2g} {unit}" for sp, x in zip(plot_data['species'], empirical_x)], hoverinfo='text'
    ))
    fig.add_trace(go.Scatter(
        x=fitted_x, y=plot_data['fitted_y_cdf_percent'], mode='lines', name=f'{dist_name} Fit', line=dict(color='#FF5252', dash='dash')
    ))
    
    p_value_percent = plot_data['p_value'] * 100
    if hcp is not None and hcp > 0:
        fig.add_trace(go.Scatter(x=[hcp], y=[p_value_percent], mode='markers', marker=dict(color='yellow', size=14, symbol='x'), name=f'HC{p_value_percent:.0f}'))
    
    fig.update_layout(
        title=f'Species Sensitivity Distribution ({dist_name} Fit)',
        xaxis_title=f'Concentration ({unit})',
        yaxis_title='Percent of Species Affected (%)',
        xaxis_type="log" if is_log_scale else "linear",
        yaxis=dict(range=[0, 100]),
        legend_title='Legend'
    )
    return fig

# --- Main Application ---
st.title("Species Sensitivity Distribution (SSD) Generator")
st.markdown("This tool generates SSD plots aligned with the **CCME Protocol for the Protection of Aquatic Life**.")

# Initialize session state for search results
if 'db_search_results' not in st.session_state:
    st.session_state.db_search_results = []

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("1. Data Source")
    data_source = st.radio("Choose data source:", ("Upload File", "Database Search"), key="data_source", horizontal=True)
    
    selected_chemicals = []
    
    if data_source == "Upload File":
        uploaded_file = st.file_uploader("Upload ecotoxicity data (CSV)", type=["csv"])
        if uploaded_file:
            chem_options = get_chemical_options_from_file(uploaded_file)
            selected_chemicals = st.multiselect("Select Chemicals from File", options=chem_options, help="Choose one or more chemicals to analyze.")
    else: 
        if supabase_conn:
            search_term = st.text_input("Search for a chemical in the database")
            if st.button("Search Database"):
                st.session_state.db_search_results = search_chemicals_in_db(supabase_conn, search_term)

            selected_chemicals = st.multiselect(
                "Select Chemicals from Search Results", 
                options=st.session_state.db_search_results,
                help="Select one or more chemicals to include in the analysis."
            )
        else:
            st.error("Supabase connection failed. Database search is unavailable.")

    st.subheader("2. Guideline & Filter Options")
    guideline_type = st.selectbox("Guideline Exposure Term", ("Long-Term", "Short-Term"))

    water_type = st.radio(
        "Water Type", ('Freshwater (FW)', 'Marine Water (MW)', 'Both'),
        horizontal=True, help="Filter data for a specific water environment. Requires a 'media_type' column in your data."
    )

    st.subheader("3. SSD Parameters")
    data_handling = st.radio(
        "Handle Multiple Values per Species", ('Use Geometric Mean', 'Use Most Sensitive (Minimum Value)'),
        horizontal=True, help="Choose how to aggregate multiple toxicity values for the same species."
    )
    distribution_fit = st.selectbox(
        "Distribution for Fitting", ('Log-Normal', 'Log-Logistic', 'Weibull'),
        help="Statistical distribution to fit to the data."
    )
    min_species = st.number_input("Minimum number of species", min_value=3, value=8, step=1)
    required_taxa = st.multiselect(
        "Required Taxonomic Groups", options=list(TAXONOMIC_MAPPING.keys()),
        default=['Fish', 'Invertebrate', 'Plant']
    )
    
    st.subheader("4. Protection & Safety")
    apply_protection_clause = st.checkbox("Apply Protection Clause", value=True)
    hcp_percentile = st.number_input("Hazard Concentration (HCp) Percentile", min_value=1.0, max_value=50.0, value=5.0, step=0.5, format="%.1f")

    generate_button = st.button("Generate SSD", type="primary", use_container_width=True)

# --- Main Processing Area ---
if generate_button:
    if not selected_chemicals:
        st.warning("Please select at least one chemical to analyze.")
        st.stop()

    df = None
    try:
        if data_source == "Upload File":
            if not uploaded_file:
                st.warning("Please upload a file to proceed.")
                st.stop()
            df = pd.read_csv(uploaded_file)
        else:
            with st.spinner(f"Fetching data for {len(selected_chemicals)} chemical(s) from Supabase..."):
                response = supabase_conn.table("toxicology_data").select("*").in_("chemical_name", selected_chemicals).execute()
                if not response.data:
                    st.error("Could not retrieve data for the selected chemicals from the database.")
                    st.stop()
                df = pd.DataFrame(response.data)

        with st.spinner("Processing data according to CCME protocol..."):
            
            # --- Start of unified processing pipeline ---
            proc_df = df[df['chemical_name'].isin(selected_chemicals)].copy()

            if 'media_type' in proc_df.columns and water_type != 'Both':
                type_code = 'FW' if 'FW' in water_type else 'MW'
                proc_df = proc_df[proc_df['media_type'] == type_code]

            if proc_df.empty:
                st.error("No data found for the selected filters. Please check your selections and data content.")
                st.stop()
            
            proc_df['conc1_mean'] = pd.to_numeric(proc_df['conc1_mean'], errors='coerce')
            proc_df.dropna(subset=['conc1_mean', 'species_scientific_name', 'endpoint'], inplace=True)
            proc_df['broad_group'] = proc_df['species_group'].apply(map_taxonomic_group)

            # This dataframe has one row for each species' most sensitive endpoint
            if guideline_type == "Long-Term":
                proc_df['endpoint_rank'] = proc_df['endpoint'].apply(get_endpoint_rank)
                proc_df.sort_values(by=['species_scientific_name', 'endpoint_rank'], ascending=True, inplace=True)
                representative_data = proc_df.drop_duplicates(subset='species_scientific_name', keep='first')
            else: # Short-term uses all valid endpoints
                representative_data = proc_df
            
            # Now, aggregate the values based on the representative data
            if data_handling == 'Use Geometric Mean':
                # Calculate geomeans for each species from the representative dataset
                geomean_values = representative_data.groupby('species_scientific_name')['conc1_mean'].apply(
                    lambda x: np.exp(np.mean(np.log(x[x > 0]))) if not x[x > 0].empty else 0
                ).reset_index()
                geomean_values.rename(columns={'conc1_mean': 'final_value'}, inplace=True)

                # Get a single row of source info for each species
                source_info = representative_data.drop_duplicates('species_scientific_name', keep='first')
                
                # Merge the calculated geomean back with the source info
                final_agg_data = pd.merge(source_info.drop(columns=['conc1_mean']), geomean_values, on='species_scientific_name')
                final_agg_data.rename(columns={'final_value': 'conc1_mean'}, inplace=True)

            else: # Use Most Sensitive (Minimum Value)
                final_agg_data = representative_data.loc[representative_data.groupby('species_scientific_name')['conc1_mean'].idxmin()]

            # --- Requirement Checks and SSD Calculation ---
            species_count = final_agg_data['species_scientific_name'].nunique()
            if species_count < min_species:
                st.error(f"Data requirements not met: Insufficient species ({species_count}). Minimum required: {min_species}.")
                st.stop()
            
            missing_taxa = [t for t in required_taxa if t not in final_agg_data['broad_group'].unique()]
            if missing_taxa:
                st.error(f"Data requirements not met: Missing required taxonomic groups: {', '.join(missing_taxa)}.")
                st.stop()
            st.success("Data requirements for Type A guideline are met.")

            hcp, params, plot_data, error_msg = calculate_ssd(
                data=final_agg_data, species_col='species_scientific_name', value_col='conc1_mean',
                p_value=hcp_percentile / 100, dist_name=distribution_fit
            )
            
            if error_msg:
                st.error(f"SSD Calculation Error: {error_msg}")
                st.stop()
            
            final_guideline = hcp
            if apply_protection_clause and not final_agg_data.empty:
                lowest_value = final_agg_data[final_agg_data['conc1_mean'] > 0]['conc1_mean'].min()
                if pd.notna(lowest_value) and lowest_value < hcp:
                    final_guideline = lowest_value
                    st.info(f"🛡️ **Protection Clause Applied:** The final guideline was set to the lowest observed data point ({lowest_value:.4g} mg/L), as it was more protective than the calculated HC{hcp_percentile:.1f}.")

            st.header("📈 Results")
            col1, col2 = st.columns(2)
            col1.metric(label=f"Calculated HC{hcp_percentile:.1f}", value=f"{hcp:.4g} mg/L")
            col2.metric(label="Final Recommended Guideline", value=f"{final_guideline:.4g} mg/L")

            ssd_fig = create_ssd_plot(plot_data, final_guideline, 'mg/L')
            st.plotly_chart(ssd_fig, use_container_width=True)
            
            # NEW/MODIFIED: Display the final data with all source columns
            st.write("#### Final Data Used for SSD Calculation")
            
            source_cols_to_display = [
                'species_scientific_name', 'broad_group', 'conc1_mean', 'endpoint',
                'reference_db', 'reference_type', 'publication_year', 
                'author', 'title', 'source', 'doi', 'source_url', 'retrieval_date', 
                'original_source', 'needs_review', 'validation_errors'
            ]
            
            # Filter to only show columns that actually exist in the final dataframe
            display_df = final_agg_data[[col for col in source_cols_to_display if col in final_agg_data.columns]].copy()
            display_df.rename(columns={'conc1_mean': 'Value (mg/L)'}, inplace=True)
            st.dataframe(display_df)

    except Exception as e:
        st.error("An unexpected error occurred during processing:")
        st.exception(e)