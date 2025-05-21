# File updated to trigger commit (no functional change)
import streamlit as st 

# --- Table Name Constants ---
TABLE_CHEMICALS = "toxicology_data"

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from io import StringIO
import supabase
from st_supabase_connection import SupabaseConnection
import plotly.express as px
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set page configuration first
st.set_page_config(layout="wide")


def initialize_supabase_connection():
    """Initialize and test Supabase connection with proper error handling.
    Returns:
        Supabase client: The initialized Supabase connection or None if failed
    """
    try:
        supabase_url = None
        supabase_key = None
        source = None
        # 1. Try Streamlit secrets [connections.supabase]
        if hasattr(st, "secrets"):
            connections_supabase = st.secrets.get("connections", {}).get("supabase", {})
            if connections_supabase.get("url") and connections_supabase.get("key"):
                supabase_url = connections_supabase.get("url")
                supabase_key = connections_supabase.get("key")
                source = "[connections.supabase]"
        # 2. Try Streamlit secrets [supabase]
        if (not supabase_url or not supabase_key) and hasattr(st, "secrets"):
            supabase_secrets = st.secrets.get("supabase", {})
            if supabase_secrets.get("url") and supabase_secrets.get("anon_key"):
                supabase_url = supabase_secrets.get("url")
                supabase_key = supabase_secrets.get("anon_key")
                source = "[supabase]"
        # 3. Try environment variables
        if (not supabase_url or not supabase_key):
            import os
            env_url = os.environ.get("SUPABASE_URL")
            env_key = os.environ.get("SUPABASE_KEY")
            if env_url and env_key:
                supabase_url = env_url
                supabase_key = env_key
                source = "environment variables"
        # Debug info (mask key)
        if supabase_url and supabase_key:
            pass

        # If still missing, error
        if not supabase_url or not supabase_key:
            st.error("Supabase configuration not found. Please add your credentials to Streamlit secrets or environment variables.")
            st.error("Checked sources in order: [connections.supabase], [supabase], environment variables.")
            st.error("Example [supabase] section in .streamlit/secrets.toml:")
            st.code("""[supabase]\nurl = \"https://your-project.supabase.co\"\nanon_key = \"your-anon-key-here\"\n""", language="toml")
            st.error("Example [connections.supabase] section in .streamlit/secrets.toml:")
            st.code("""[connections.supabase]\nurl = \"https://your-project.supabase.co\"\nkey = \"your-anon-key-here\"\n""", language="toml")
            st.error("Or set SUPABASE_URL and SUPABASE_KEY as environment variables.")
            return None
        from supabase import create_client
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        st.error(f"Failed to initialize Supabase connection: {e}")
        return None
        # If not found in secrets, try environment variables (for local development)
        if not supabase_url or not supabase_key:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            st.error("Supabase credentials not found")
            st.error("Please configure these in either:")
            st.error("1. Streamlit Cloud Secrets (recommended for deployment):")
            st.error("   - Go to your Streamlit app settings")
            st.error("   - Add a new section called [supabase]")
            st.error("   - Add the following keys:")
            st.error("     - url: Your Supabase project URL")
            st.error("     - anon_key: Your Supabase anon key")
            st.error("""Example configuration:
[supabase]
url = "https://your-project.supabase.co"
anon_key = "your-anon-key-here"
""")
            st.error("OR")
            st.error("2. Environment variables (for local development):")
            st.error("   - Create a .env file in your project root")
            st.error("   - Add SUPABASE_URL and SUPABASE_KEY")
            return None

        from supabase import create_client
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        st.error(f"Failed to initialize Supabase connection: {e}")
        return None

supabase_conn = initialize_supabase_connection()
if supabase_conn is None:
    st.error("Supabase connection could not be established. Please check your credentials and setup.")
    st.stop()

# Initialize session state
if 'chemicals_loaded' not in st.session_state:
    st.session_state.chemicals_loaded = False
if 'chemicals_data' not in st.session_state:
    st.session_state.chemicals_data = []

# --- Supabase connection test temporarily disabled ---
# try:
#     # Try to fetch a row from the toxicology_data table to verify connection
#     test_query = supabase_conn.table("toxicology_data").select("id, chemical_name").limit(1).execute()
#     if test_query.data:
#         st.success("Successfully connected to Supabase!")
#         st.write("Sample data from 'toxicology_data':", test_query.data)
#     else:
#         st.error("Unable to access any data from the 'toxicology_data' table.")
#         st.error("Please check:")
#         st.error("1. The anon key has the correct permissions")
#         st.error("2. RLS policies are correctly configured")
#         st.error("3. The 'toxicology_data' table exists and is not empty.")
#         raise Exception("No data accessible with current credentials")
#
# except Exception as e:
#     st.error(f"Error testing connection to Supabase: {str(e)}")
#     st.error("Please verify your Supabase credentials, permissions, and table existence.")
#     import traceback
#     st.error(traceback.format_exc())


# try:
#     # Try to fetch a row from the toxicology_data table to verify connection (using valid columns)
#     test_row = supabase_conn.table(TABLE_CHEMICALS).select("id, chemical_name").limit(1).execute()
#     if not test_row.data:
#         st.error("No data found in toxicology_data table.")
#         raise Exception("No data accessible with current credentials")
#     else:
#         st.success("Supabase connection and table access verified.")
# except Exception as e:
#     st.error(f"Connection test failed: {str(e)}")
#     st.error("Please check:")
#     st.error("1. Your Supabase URL and anon key are correct")
#     st.error("2. The URL is accessible")
#     st.error("3. The anon key has the correct permissions")
#     st.error("4. The database table TABLE_CHEMICALS exists")
#     st.error("Full traceback:")
#     import traceback
#     st.error(traceback.format_exc())

# Move the rest of the configuration section back to its original position
ECOTOX_EXPECTED_COLS = {
    'cas': 'test_cas',            # CAS Registry Number
    'chemical': 'chemical_name',  # Chemical Name
    'species_sci': 'species_scientific_name', # Species Scientific Name
    'species_common': 'species_common_name', # Species Common Name (Optional)
    'group': 'species_group',     # Taxonomic Group
    'endpoint': 'endpoint',       # Endpoint code
    'effect': 'effect',           # Effect measured
    'conc_mean': 'conc1_mean',    # Mean concentration value
    'conc_unit': 'conc1_unit'     # Concentration unit
}

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

# --- Media Classification based on Units ---
MEDIA_UNITS = {
    # Water/Wastewater
    'Water/Wastewater': [
        'mg/L', 'µg/L', 'ng/L', 'pg/L',  # Concentration
        'mg/m³', 'µg/m³', 'ng/m³', 'pg/m³',  # Volume
        'ppm', 'ppb', 'ppt',  # Parts per
        'mg/kg', 'µg/kg', 'ng/kg', 'pg/kg'  # Sediment/Water
    ],
    # Soil/Sediment
    'Soil/Sediment': [
        'mg/kg', 'µg/kg', 'ng/kg', 'pg/kg',  # Concentration
        'ppm', 'ppb', 'ppt',  # Parts per
        'mg/L', 'µg/L', 'ng/L', 'pg/L',  # Pore water
        '%', 'ppt'  # Percentage
    ],
    # Air
    'Air': [
        'mg/m³', 'µg/m³', 'ng/m³', 'pg/m³',  # Concentration
        'ppm', 'ppb', 'ppt',  # Parts per
        'mg/L', 'µg/L', 'ng/L', 'pg/L',  # Volume
        '%', 'ppt'  # Percentage
    ],
    # Biota
    'Biota': [
        'mg/kg', 'µg/kg', 'ng/kg', 'pg/kg',  # Concentration
        'ppm', 'ppb', 'ppt',  # Parts per
        'mg/L', 'µg/L', 'ng/L', 'pg/L',  # Tissue
        '%', 'ppt'  # Percentage
    ],
    # Food
    'Food': [
        'mg/kg', 'µg/kg', 'ng/kg', 'pg/kg',  # Concentration
        'ppm', 'ppb', 'ppt',  # Parts per
        '%', 'ppt'  # Percentage
    ]
}

# --- Media Classification Function ---
def get_media_from_unit(unit):
    """
    Classify chemical based on its measurement unit to determine the likely media it's measured in.
    Returns the primary media if unit matches exactly, or 'Unknown' if no match.
    """
    if not unit:
        return "Unknown"
    
    # Normalize unit to lowercase for matching
    unit = unit.lower().strip()
    
    # Check each media category
    for media, units in MEDIA_UNITS.items():
        if unit in [u.lower() for u in units]:
            return media
    
    # If no exact match, try partial matches
    if any(u in unit for u in ['mg/l', 'ug/l', 'ng/l', 'pg/l', 'ppm', 'ppb', 'ppt']):
        return "Water/Wastewater"
    elif any(u in unit for u in ['mg/kg', 'ug/kg', 'ng/kg', 'pg/kg']):
        return "Soil/Sediment"
    elif any(u in unit for u in ['mg/m3', 'ug/m3', 'ng/m3', 'pg/m3']):
        return "Air"
    
    return "Unknown"

# --- Chemical Group Classification ---
def get_chemical_group(species_group):
    """
    Classify chemicals into groups based on species group and other properties.
    This is a comprehensive classification system that categorizes chemicals based on their properties.
    """
    if not species_group:
        return "Unknown"
    
    # Convert to lowercase for easier matching
    species_group = species_group.lower()
    
    # Organic Compounds
    if any(org in species_group for org in ['organic', 'carbon', 'hydrocarbon', 'alkyl', 'aryl', 'benzene']):
        if any(aliph in species_group for aliph in ['alkane', 'alkene', 'alkyne', 'aliphatic']):
            return "Organic Compounds (Aliphatic)"
        elif any(arom in species_group for arom in ['aromatic', 'benzene', 'phenyl', 'aryl']):
            return "Organic Compounds (Aromatic)"
        else:
            return "Organic Compounds (Other)"
    
    # Inorganic Compounds
    if any(inorg in species_group for inorg in ['inorganic', 'salt', 'oxide', 'hydroxide']):
        if any(acid in species_group for acid in ['acid', 'hydrochloric', 'sulfuric', 'nitric']):
            return "Inorganic Compounds (Acids)"
        elif any(base in species_group for base in ['base', 'alkali', 'hydroxide', 'ammonia']):
            return "Inorganic Compounds (Bases)"
        elif any(salt in species_group for salt in ['salt', 'chloride', 'sulfate', 'nitrate']):
            return "Inorganic Compounds (Salts)"
        else:
            return "Inorganic Compounds (Other)"
    
    # Metals
    if any(metal in species_group for metal in ['metal', 'alloy', 'oxide', 'hydroxide', 'sulfide']):
        if any(heavy in species_group for heavy in ['lead', 'mercury', 'cadmium', 'arsenic']):
            return "Metals (Heavy Metals)"
        elif any(trans in species_group for trans in ['transition', 'iron', 'copper', 'nickel']):
            return "Metals (Transition Metals)"
        elif any(alkali in species_group for alkali in ['sodium', 'potassium', 'lithium']):
            return "Metals (Alkali Metals)"
        else:
            return "Metals (Other)"
    
    # Pesticides
    if any(pest in species_group for pest in ['pesticide', 'herbicide', 'fungicide', 'insecticide', 'rodenticide']):
        if any(herb in species_group for herb in ['herbicide', 'weed', 'plant', 'weedkiller']):
            return "Pesticides (Herbicides)"
        elif any(insect in species_group for insect in ['insecticide', 'bug', 'insect', 'pest']):
            return "Pesticides (Insecticides)"
        elif any(fung in species_group for fung in ['fungicide', 'mold', 'fungus', 'mildew']):
            return "Pesticides (Fungicides)"
        else:
            return "Pesticides (Other)"
    
    # Pharmaceuticals
    if any(pharm in species_group for pharm in ['drug', 'pharmaceutical', 'medicine', 'antibiotic', 'antibacterial']):
        if any(anti in species_group for anti in ['antibiotic', 'penicillin', 'amoxicillin', 'tetracycline']):
            return "Pharmaceuticals (Antibiotics)"
        elif any(anti in species_group for anti in ['antiviral', 'virus', 'influenza', 'herpes']):
            return "Pharmaceuticals (Antivirals)"
        elif any(pain in species_group for pain in ['analgesic', 'pain', 'aspirin', 'ibuprofen', 'acetaminophen']):
            return "Pharmaceuticals (Analgesics)"
        else:
            return "Pharmaceuticals (Other)"
    
    # Plastics and Polymers
    if any(plastic in species_group for plastic in ['plastic', 'polymer', 'polyethylene', 'polypropylene', 'polyvinyl']):
        if any(thermo in species_group for thermo in ['thermoplastic', 'polyethylene', 'polypropylene', 'polyvinyl']):
            return "Plastics (Thermoplastics)"
        elif any(thermo in species_group for thermo in ['thermoset', 'epoxy', 'phenolic', 'polyurethane']):
            return "Plastics (Thermosets)"
        elif any(bio in species_group for bio in ['biodegradable', 'bioplastic', 'compostable', 'PLA']):
            return "Plastics (Biodegradable)"
        else:
            return "Plastics (Other)"
    
    # Solvents
    if any(solvent in species_group for solvent in ['solvent', 'alcohol', 'ether', 'ketone', 'ester', 'benzene']):
        if any(polar in species_group for polar in ['polar', 'water', 'alcohol', 'ether']):
            return "Solvents (Polar)"
        elif any(non in species_group for non in ['non-polar', 'hydrocarbon', 'hexane', 'benzene']):
            return "Solvents (Non-polar)"
        elif any(aprot in species_group for aprot in ['aprotic', 'dichloromethane', 'acetone', 'acetonitrile']):
            return "Solvents (Aprotic)"
        else:
            return "Solvents (Other)"
    
    # Surfactants
    if any(surf in species_group for surf in ['surfactant', 'detergent', 'emulsifier', 'tenside', 'wetting agent']):
        if any(anion in species_group for anion in ['anionic', 'sulfate', 'sulfonate', 'carboxylate']):
            return "Surfactants (Anionic)"
        elif any(cation in species_group for cation in ['cationic', 'quaternary', 'ammonium', 'amine']):
            return "Surfactants (Cationic)"
        elif any(non in species_group for non in ['non-ionic', 'alkoxylate', 'ether', 'ester']):
            return "Surfactants (Non-ionic)"
        else:
            return "Surfactants (Other)"
    
    # Dyes and Pigments
    if any(dye in species_group for dye in ['dye', 'pigment', 'colorant', 'stain', 'paint', 'ink']):
        if any(acid in species_group for acid in ['acid', 'sulfonic', 'carboxylic', 'dye']):
            return "Dyes (Acidic)"
        elif any(base in species_group for base in ['basic', 'amine', 'azo', 'dye']):
            return "Dyes (Basic)"
        elif any(direct in species_group for direct in ['direct', 'azo', 'anthraquinone', 'dye']):
            return "Dyes (Direct)"
        else:
            return "Dyes (Other)"
    
    # Industrial Chemicals
    if any(ind in species_group for ind in ['industrial', 'chemical', 'reagent', 'catalyst', 'additive', 'lubricant']):
        if any(corro in species_group for corro in ['corrosive', 'acid', 'base', 'hydrofluoric']):
            return "Industrial Chemicals (Corrosives)"
        elif any(flam in species_group for flam in ['flammable', 'solvent', 'fuel', 'petroleum']):
            return "Industrial Chemicals (Flammables)"
        elif any(tox in species_group for tox in ['toxic', 'poison', 'cyanide', 'mercury']):
            return "Industrial Chemicals (Toxic)"
        else:
            return "Industrial Chemicals (Other)"
    
    # Nanomaterials
    if any(nano in species_group for nano in ['nano', 'nanoparticle', 'nanomaterial', 'quantum dot', 'carbon nanotube']):
        if any(metal in species_group for metal in ['metallic', 'gold', 'silver', 'copper']):
            return "Nanomaterials (Metallic)"
        elif any(carbon in species_group for carbon in ['carbon', 'graphene', 'nanotube', 'fullerene']):
            return "Nanomaterials (Carbon-based)"
        elif any(oxide in species_group for oxide in ['oxide', 'silica', 'titanium', 'zinc']):
            return "Nanomaterials (Oxide)"
        else:
            return "Nanomaterials (Other)"
    
    # Radioactive Substances
    if any(radio in species_group for radio in ['radioactive', 'isotope', 'radiation', 'nuclear']):
        if any(alpha in species_group for alpha in ['alpha', 'helium', 'radon', 'polonium']):
            return "Radioactive Substances (Alpha)"
        elif any(beta in species_group for beta in ['beta', 'electron', 'strontium', 'cesium']):
            return "Radioactive Substances (Beta)"
        elif any(gamma in species_group for gamma in ['gamma', 'photon', 'cobalt', 'iridium']):
            return "Radioactive Substances (Gamma)"
        else:
            return "Radioactive Substances (Other)"
    
    # Biocides
    if any(bio in species_group for bio in ['biocide', 'disinfectant', 'antimicrobial', 'preservative', 'fungicide']):
        if any(disin in species_group for disin in ['disinfectant', 'sterilizer', 'bleach', 'alcohol']):
            return "Biocides (Disinfectants)"
        elif any(anti in species_group for anti in ['antimicrobial', 'antibacterial', 'antifungal', 'antiviral']):
            return "Biocides (Antimicrobials)"
        elif any(pres in species_group for pres in ['preservative', 'fungicide', 'mold', 'rot']):
            return "Biocides (Preservatives)"
        else:
            return "Biocides (Other)"
    
    # Food Additives
    if any(food in species_group for food in ['food', 'additive', 'preservative', 'emulsifier', 'stabilizer', 'flavor']):
        if any(pres in species_group for pres in ['preservative', 'sodium', 'benzoate', 'sorbate']):
            return "Food Additives (Preservatives)"
        elif any(color in species_group for color in ['colorant', 'dye', 'pigment', 'food color']):
            return "Food Additives (Colorants)"
        elif any(emul in species_group for emul in ['emulsifier', 'stabilizer', 'lethicin', 'gum']):
            return "Food Additives (Emulsifiers)"
        else:
            return "Food Additives (Other)"
    
    # Cosmetics
    if any(cos in species_group for cos in ['cosmetic', 'personal care', 'skin care', 'hair care', 'makeup']):
        if any(skin in species_group for skin in ['skin', 'cream', 'lotion', 'serum', 'moisturizer']):
            return "Cosmetics (Skin Care)"
        elif any(hair in species_group for hair in ['hair', 'shampoo', 'conditioner', 'dye', 'gel']):
            return "Cosmetics (Hair Care)"
        elif any(make in species_group for make in ['makeup', 'foundation', 'lipstick', 'eyeshadow', 'mascara']):
            return "Cosmetics (Makeup)"
        else:
            return "Cosmetics (Other)"
    
    # Default to Organic Compounds if no specific category matches
    return "Organic Compounds (Other)"

# --- Helper Functions --- (Keep these as is: map_taxonomic_group, get_distribution, calculate_ssd, create_ssd_plot)

def get_chemical_options(uploaded_file):
    """
    Reads the uploaded file and returns a list of chemical options.
    """
    import pandas as pd
    from io import StringIO

    if uploaded_file is None:
        return ["-- Upload File First --"]
    try:
        uploaded_file.seek(0)
        file_content = uploaded_file.getvalue()
        # Try CSV first
        try:
            df = pd.read_csv(StringIO(file_content.decode('latin-1')), sep=',')
        except Exception:
            df = pd.read_csv(StringIO(file_content.decode('latin-1')), sep='\t')
        # Use the expected chemical name column
        chem_col = ECOTOX_EXPECTED_COLS.get('chemical', 'chemical_name')
        return get_chemical_names(df, chem_col)
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        return ["-- Error Reading File --"]

def map_taxonomic_group(ecotox_group):
    """Maps detailed ECOTOX group to broader category."""
    for broad_group, detailed_list in TAXONOMIC_MAPPING.items():
        if ecotox_group in detailed_list:
            return broad_group
    return "Other"

def get_distribution(dist_name):
    """Returns the distribution parameters based on name."""
    if dist_name == 'lognormal':
        return 'lognormal'
    elif dist_name == 'normal':
        return 'normal'
    elif dist_name == 'weibull':
        return 'weibull'
    return 'lognormal'

def calculate_ssd(data, species_col, value_col, dist_name, p_value):
    """Calculates SSD parameters and HCp using numpy."""
    # Diagnostic output for debugging
    st.write("SSD calculation input summary:")
    st.write(data[value_col].describe())
    st.write("Min value:", data[value_col].min())
    # Check for positive concentrations
    valid_data = data[data[value_col] > 0].copy()
    if valid_data.empty:
        return None, None, None, "No positive concentrations for SSD calculation."
    # Check p_value
    if not (0 < p_value < 1):
        return None, None, None, f"p_value must be between 0 and 1 (got {p_value})."
    try:
        # Calculate parameters using numpy
        if dist_name == 'lognormal':
            # For lognormal, we need to transform data
            log_data = np.log(data[value_col])
            mean = np.mean(log_data)
            std = np.std(log_data)
            median = np.exp(mean)
            # Calculate HCP using inverse CDF (correct lognormal quantile)
            hcp = np.exp(mean + std * norm.ppf(p_value))
            params = (mean, std)
        elif dist_name == 'normal':
            mean = np.mean(data[value_col])
            std = np.std(data[value_col])
            median = mean
            # Calculate HCP using inverse CDF (correct normal quantile)
            hcp = mean + std * norm.ppf(p_value)
            params = (mean, std)
        else:  # normal or weibull
            # Fit in log space for alignment with plotting
            log_values = np.log(data[value_col])
            mean = np.mean(log_values)
            std = np.std(log_values)
            median = np.exp(mean)
            # Calculate HCP using inverse CDF in log space (for loglogistic/weibull, fallback to lognormal formula)
            hcp = np.exp(mean + std * norm.ppf(p_value))
            params = (mean, std)
        
        # Compose plot_data for SSD plotting
        # Empirical CDF (convert to log10 scale for plotting)
        log_values = np.log(data[value_col])
        log10_values = log_values / np.log(10)
        sorted_log10_values = np.sort(log10_values)
        empirical_cdf = np.arange(1, len(sorted_log10_values) + 1) / (len(sorted_log10_values) + 1)
        # Fitted CDF (use norm.cdf for lognormal fit)
        # Generate log10 x values over the observed range
        min_x = np.floor(sorted_log10_values.min())
        max_x = np.ceil(sorted_log10_values.max())
        fitted_log10_values = np.linspace(min_x, max_x, 200)
        # Convert log10 x to ln x for the CDF
        fitted_ln_values = fitted_log10_values * np.log(10)
        # CDF for lognormal: use normal CDF in ln space
        fitted_cdf = norm.cdf(fitted_ln_values, loc=mean, scale=std)
        fitted_cdf_percent = fitted_cdf * 100
        log_hcp = np.log(hcp) if hcp > 0 else float('nan')
        log10_hcp = log_hcp / np.log(10) if hcp > 0 else float('nan')
        plot_data = {
            'empirical_log_values': sorted_log10_values,
            'empirical_cdf_percent': empirical_cdf * 100,
            'fitted_log_values': fitted_log10_values,
            'fitted_cdf_percent': fitted_cdf_percent,
            'log_hcp': log10_hcp,
            'hcp_p_percent': p_value,
            'species': data[species_col].tolist() if species_col in data else []
        }
        return hcp, params, plot_data, None
    except Exception as e:
        st.error(f"Error calculating SSD: {str(e)}")
        return None, None, None, str(e)

# --- SSD Plotting Function ---
def create_ssd_plot(plot_data, hcp, p_value, dist_name, unit):
    """ Generates the SSD Plotly figure with x-axis in real concentration units (log scale). """
    if plot_data is None: return go.Figure()
    import numpy as np
    import plotly.graph_objects as go
    # Transform log10 values back to real concentrations for plotting
    empirical_x = 10 ** np.array(plot_data['empirical_log_values'])
    fitted_x = 10 ** np.array(plot_data['fitted_log_values'])
    hcp_x = 10 ** plot_data['log_hcp'] if plot_data['log_hcp'] is not None and not np.isnan(plot_data['log_hcp']) else None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=empirical_x, y=plot_data['empirical_cdf_percent'], mode='markers', name='Species Data',
        marker=dict(color='#2196F3', size=8, symbol='circle', opacity=0.85),
        hovertext=[f"Species: {sp}<br>Conc: {x:.2g} {unit}" for sp, x in zip(plot_data['species'], empirical_x)], hoverinfo='text'
    ))
    fig.add_trace(go.Scatter(
        x=fitted_x, y=plot_data['fitted_cdf_percent'], mode='lines', name=f'Fitted {dist_name} CDF', line=dict(color='#FF5252', dash='dash', width=2)
    ))
    # Mark the HCp point if valid
    if hcp_x is not None and hcp_x > 0:
        fig.add_trace(go.Scatter(x=[hcp_x], y=[p_value*100], mode='markers', marker=dict(color='#FFD600', size=14, symbol='x', opacity=1), name=f'HC{p_value}'))
    # Compute axis range to include all points and HCp
    all_x = np.concatenate([empirical_x, fitted_x, [hcp_x] if hcp_x is not None and hcp_x > 0 else []])
    all_x = all_x[all_x > 0]  # Remove non-positive values for log scale
    if len(all_x) > 0:
        xmin = np.nanmin(all_x)
        xmax = np.nanmax(all_x)
        xmargin = (np.log10(xmax) - np.log10(xmin)) * 0.1 if xmax > xmin else 1
        fig.update_xaxes(type="log", range=[np.log10(xmin) - xmargin, np.log10(xmax) + xmargin])
    else:
        fig.update_xaxes(type="log")
    fig.update_layout(
        title='Species Sensitivity Distribution (SSD)',
        title_x=0.5,
        title_font_size=18,
        xaxis_title=f'Concentration ({unit})',
        xaxis_title_font_size=14,
        yaxis_title='Percent of Species Affected (%)',
        yaxis_title_font_size=14,
        legend_title='Legend',
        legend_title_font_size=14,
        legend_font_size=12,
        yaxis=dict(range=[0, 100], gridwidth=1, gridcolor='#444', color='#FFF'),
        xaxis=dict(gridwidth=1, gridcolor='#444', color='#FFF',
                   tickvals=[0.01, 0.1, 1, 10, 100, 1000, 10000],
                   ticktext=['0.01', '0.1', '1', '10', '100', '1000', '10000']),
        hovermode='closest',
        plot_bgcolor='#111',
        paper_bgcolor='#000',
        font=dict(color='#FFF'),

    )
    # Add HCp marker if it is valid (only ONCE)
    if hcp_x is not None:
        hcp_percent = plot_data.get('hcp_p_percent', p_value)
        if hcp_percent <= 1.0:
            hcp_percent = hcp_percent * 100
        fig.add_trace(go.Scatter(
            x=[hcp_x],
            y=[hcp_percent],
            mode='markers',
            marker=dict(size=14, color='yellow', symbol='x'),
            name=f'HC{hcp_percent:.1f}'
        ))
    # Hide duplicate HCp legend entries
    seen_names = set()
    for trace in fig.data:
        if hasattr(trace, 'name') and trace.name in seen_names and trace.name and trace.name.startswith("HC"):
            trace.showlegend = False
        else:
            seen_names.add(getattr(trace, 'name', None))
    return fig

def get_chemical_names(df_chem, chem_col):
    """Extracts and returns a sorted list of unique chemical names from the specified column in df_chem."""
    try:
        if chem_col in df_chem.columns:
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


def initialize_supabase_connection():
    """Initialize and test Supabase connection with proper error handling.
    Returns:
        SupabaseConnection: The initialized Supabase connection or None if failed
    """
    try:
        # First try to get credentials from Streamlit secrets (for Streamlit Cloud)
        supabase_config = st.secrets.get("connections", {}).get("supabase", {})
        
        # Get URL and key from config
        supabase_url = supabase_config.get("url")
        supabase_key = supabase_config.get("key") # anon_key
        
        # If not found in secrets, try environment variables (for local development)
        if not supabase_url or not supabase_key:
            # Try environment variables
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_KEY")

        if not supabase_url or not supabase_key:
            raise ValueError("Supabase configuration not found. Please configure your credentials in Streamlit secrets or environment variables.")

        # Create Supabase connection
        try:
            supabase_conn = st.connection(
                "supabase",
                type=SupabaseConnection,
                url=supabase_url,
                key=supabase_key
            )
            
            # Test the connection
            try:
                # Try a simple query to test the connection
                test_result = supabase_conn.table(TABLE_CHEMICALS).select("chemical_name").limit(1).execute()
                if len(test_result.data) > 0:
                    st.success("Successfully connected to Supabase!")
                    return supabase_conn
                else:
                    raise ValueError("Supabase connection successful but toxicology_data table not found")
            except Exception as e:
                raise ValueError(f"Failed to test Supabase connection: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to create Supabase connection: {str(e)}")

    except Exception as e:
        st.error(f"Failed to initialize Supabase connection: {str(e)}")
        raise e

# Initialize session state
if 'chemicals_loaded' not in st.session_state:
    st.session_state.chemicals_loaded = False
    st.session_state.chemicals_data = []

# Initialize Supabase connection
supabase_conn = initialize_supabase_connection()

# If connection failed, don't proceed with the rest of the app
if supabase_conn is None:
    st.stop()

def fetch_chemicals(search_term=None):
    chemical_groups = {}
    chem_data = []
    if not search_term or not search_term.strip():
        st.warning("Please enter a search term to fetch chemicals. Fetching all records is disabled to avoid timeouts.")
        st.session_state.chemical_groups = chemical_groups  # Always set in session state
        return False
    """Fetch chemicals from Supabase and process them.
    

    """
    try:
        # Use ilike for wildcard, case-insensitive search if search_term is provided
        # Only select required columns and limit the number of records to avoid timeouts
        columns = "id, test_cas, chemical_name, species_scientific_name, species_common_name, species_group, endpoint, effect, conc1_mean, conc1_unit"
        query = supabase_conn.table("toxicology_data").select(columns)
        if search_term and search_term.strip():
            # Substring search using trigram index
            query = query.ilike("chemical_name", f"%{search_term.strip()}%")
        query = query.limit(1000)
        chemicals = query.execute()
        
        # Check if we got any data
        if not chemicals.data:
            st.error("No records found in the toxicology_data table")
            st.session_state.chemical_groups = chemical_groups
            return False
            
        # Convert to DataFrame
        df = pd.DataFrame(chemicals.data)
        

        # Display grouped summary by chemical_name
        grouped = df.groupby('chemical_name').size().reset_index(name='count')
        st.write('Grouped Results by Chemical Name:')
        st.dataframe(grouped)
        
        # Add chemical group column
        if 'chemical_name' in df.columns:
            df['group'] = df['chemical_name'].apply(get_chemical_group)
        else:
            df['group'] = 'Unknown'
        
        # Add media classification based on units (if conc1_unit exists)
        if 'conc1_unit' in df.columns:
            df['media'] = df['conc1_unit'].apply(get_media_from_unit)
        else:
            df['media'] = 'Unknown'
        
        # Add occurrence count
        df['occurrences'] = 1
        
        # Ensure test_cas is a string
        if 'test_cas' in df.columns:
            df['test_cas'] = df['test_cas'].astype(str)
            # Clean up CAS numbers by removing any non-digit characters
            df['test_cas'] = df['test_cas'].str.replace(r'[^\d-]', '', regex=True)
            # Handle empty or invalid CAS numbers
            df['test_cas'] = df['test_cas'].apply(lambda x: x if x and len(x) > 0 else "")
        
        # Populate chemical_groups from the DataFrame
        if 'group' in df.columns:
            chemical_groups = df['group'].value_counts().to_dict()
        else:
            chemical_groups = {}
        st.session_state.chemical_groups = chemical_groups

        # Store in session state
        st.session_state.chemicals_data = df.to_dict('records')
        st.session_state.chemicals_loaded = True
        
        return True
    except Exception as e:
        st.error(f"Failed to fetch records from toxicology_data: {str(e)}")
        st.session_state.chemical_groups = chemical_groups  # Ensure always defined
        st.exception(e)
        return False

# Using pre-initialized Supabase connection from initialization section
if supabase_conn:
    # [REMOVED: Legacy main-panel chemical management expander]
    # All chemical management UI is now in the sidebar.
    # [REMOVED: orphaned block after expander deletion]
    # All chemical management UI is now in the sidebar.
    pass


# Display all chemicals fetched from Supabase
if st.session_state.chemicals_loaded and st.session_state.chemicals_data:
    st.write("### Chemicals Fetched from Supabase")
    chem_df = pd.DataFrame(st.session_state.chemicals_data)
    st.dataframe(chem_df, hide_index=True)
    # Add download button for full fetched data
    csv = chem_df.to_csv(index=False)
    st.download_button(
        label="Download All Fetched Data as CSV",
        data=csv,
        file_name="supabase_chemical_data.csv",
        mime="text/csv",
        key="download_supabase_chemicals_csv"
    )

# Show search results
group_options = ['All']  # Ensure group_options is always defined
if st.session_state.chemicals_loaded:
    # Filter by search term
    filtered_chems = st.session_state.chemicals_data

    
    # Filter by groups
    if 'All' not in group_options:
        filtered_chems = [chem for chem in filtered_chems 
                        if chem['group'] in group_options]
    
    if filtered_chems:
        st.write(f"Found {len(filtered_chems)} matching chemicals:")
        chem_df = pd.DataFrame(filtered_chems)
        
        # Add multi-select for chemicals
        selected_chemicals = st.multiselect(
            "Select Chemicals",
            options=chem_df['chemical_name'].tolist(),
            help="Select multiple chemicals by holding Ctrl/Cmd"
        )
    else:
        st.info("No chemicals found matching your search.")

    if selected_chemicals:
        # Show selected chemicals
            selected_df = chem_df[chem_df['chemical_name'].isin(selected_chemicals)]
            st.write("Selected Chemicals:")
            st.dataframe(selected_df, hide_index=True)
            
            # Add download option for selected chemicals
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Download Selected Chemicals", key="download_selected_btn_1"):
                    csv = selected_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="selected_chemicals.csv",
                        mime="text/csv",
                        key="download_selected_csv_1"
                    )
            with col2:
                if st.button("Download Complete List", key="download_complete_btn_1"):
                    chem_df = pd.DataFrame(st.session_state.chemicals_data)
                    csv = chem_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="complete_chemical_list.csv",
                        mime="text/csv",
                        key="download_complete_csv_1"
                    )
else:
    st.info("No chemicals found matching your filters.")

# Add chemical count 
if st.session_state.chemicals_loaded:
    chem_count = len(st.session_state.chemicals_data)
    st.write(f"Total unique chemicals in database: {chem_count}")

# --- Initialize Session State ---
# Use session state to prevent resetting dropdown when other widgets change
if 'selected_chemicals' not in st.session_state:
    st.session_state.selected_chemicals = []
if 'file_processed_chem_list' not in st.session_state:
    st.session_state.file_processed_chem_list = None

# --- Sidebar for Inputs ---
# --- Sidebar: Single source of truth for file upload and chemical selection ---
with st.sidebar:
    uploaded_file = st.file_uploader("Import CSV or TXT", type=["csv", "txt"], help="Upload your chemical data file.", key="file_upload")
    st.markdown("---")
    st.markdown("### Chemical Management")
    if uploaded_file is not None:
        if uploaded_file != st.session_state.get('last_uploaded_file', None) or st.session_state.file_processed_chem_list is None:
            st.session_state.last_uploaded_file = uploaded_file
            with st.spinner("Reading chemical list..."):
                st.session_state.file_processed_chem_list = get_chemical_options(uploaded_file)
        chemical_options = st.session_state.file_processed_chem_list or ["-- Error Reading File --"]
        # --- FILE UPLOADED: Only show file-based workflow ---
        st.markdown("#### Chemical Selection (From Uploaded File)")
        st.info("You have uploaded a file. The chemical selection and options below use only your uploaded data. To use the database, remove the file.")
        key_suffix = '_file'
        current_chemical_options = chemical_options
        selected_chemicals = st.multiselect(
            "Select Chemicals from File",
            options=current_chemical_options,
            key="selected_chemicals_file",
            help="Hold Ctrl/Cmd or use checkboxes to select multiple chemicals. Start typing to filter."
        )
        # Filter out the placeholder
        selected_chemicals = [c for c in selected_chemicals if c != "-- Select Chemical --"]
        valid_chem_names = [opt for opt in current_chemical_options if not opt.startswith('--') and opt.strip()]
        if len(valid_chem_names) == 0:
            st.warning("No valid chemical names found. Please check your file format.")
        endpoint_type = st.radio(
            "Endpoint Type", ('Acute (LC50, EC50)', 'Chronic (NOEC, LOEC, EC10)'), index=0,
            key="endpoint_type_file",
            help="Select the general type of endpoint to include."
        )
        min_species = st.number_input(
            "Minimum Number of Species", min_value=3, value=5, step=1,
            key=f"min_species{key_suffix}",
            help="Minimum unique species required after filtering."
        )
        required_taxa_broad = st.multiselect(
            "Required Taxonomic Groups", options=list(TAXONOMIC_MAPPING.keys()), default=list(TAXONOMIC_MAPPING.keys())[:3],
            key=f"required_taxa_broad{key_suffix}",
            help="Select the broad taxonomic groups that *must* be represented."
        )
        data_handling = st.radio(
            "Handle Multiple Values per Species", ('Use Geometric Mean', 'Use Most Sensitive (Minimum Value)'), index=0,
            key=f"data_handling{key_suffix}",
        )
        # *** MODIFIED: Button enabling logic for multi-select ***
        is_ready_to_generate = (
            uploaded_file is not None and
            selected_chemicals and
            all([c not in (None, "-- Upload File First --", "-- Error Reading File --") for c in selected_chemicals])
        )
        generate_button = st.button("🚀 Generate SSD", disabled=(not is_ready_to_generate))

# All downstream logic should use the sidebar's uploaded_file, chemical_options, and selected_chemicals

        if len(valid_chem_names) == 0:
            st.warning("No valid chemical names found. Please check your file format.")
        endpoint_type = st.radio(
            "Endpoint Type", ('Acute (LC50, EC50)', 'Chronic (NOEC, LOEC, EC10)'), index=0,
            key="downstream_endpoint_type",
            help="Select the general type of endpoint to include."
        )
        min_species = st.number_input(
            "Minimum Number of Species", min_value=3, value=5, step=1,
            key="downstream_min_species",
            help="Minimum unique species required after filtering."
        )
        required_taxa_broad = st.multiselect(
            "Required Taxonomic Groups", options=list(TAXONOMIC_MAPPING.keys()), default=list(TAXONOMIC_MAPPING.keys())[:3],
            key="downstream_required_taxa_broad",
            help="Select the broad taxonomic groups that *must* be represented."
        )
        data_handling = st.radio(
            "Handle Multiple Values per Species", ('Use Geometric Mean', 'Use Most Sensitive (Minimum Value)'), index=0,
            key="downstream_data_handling",
            help="How to aggregate multiple data points for the same species."
        )
    else:
        # --- NO FILE: Always show Supabase search/filter/fetch UI ---
        with st.sidebar:
            st.markdown("#### Chemical Search and Filters (From Database)")
            st.info("No file uploaded. The options below let you search and filter chemicals from the central database.")
            key_suffix = '_supabase'
            search_term = st.text_input(
                "Search Toxicology Data",
                key=f"chem_search{key_suffix}",
                help="You can now search for any part of a chemical name (e.g., 'ace' will match 'Acetone'). Enter at least 3 characters."
            )
            if search_term and len(search_term.strip()) < 3:
                st.warning("Please enter at least 3 characters to search any part of the chemical name.")
                search_term = None
            # Fetch button directly under search
            if st.button("Fetch Toxicology Data from Supabase", key="fetch_chemicals_btn_sidebar"):
                try:
                    with st.spinner("Fetching chemical list from Supabase..."):
                        if fetch_chemicals(search_term=search_term):
                            st.success("Successfully fetched chemicals!")
                except Exception as e:
                    st.error(f"Failed to fetch records from toxicology_data: {str(e)}")
                    st.exception(e)
            group_options = st.multiselect(
                "Filter by Group",
                options=["All"] + sorted(set([chem.get('group', 'Unknown') for chem in st.session_state.get('chemicals_data', [])])),
                default=["All"],
                key=f"group_filter{key_suffix}",
                help="Select chemical groups to filter the search results"
            )
            media_options = st.multiselect(
                "Filter by Media",
                options=['All', 'Water/Wastewater', 'Soil/Sediment', 'Air', 'Biota', 'Food'],
                default=['All'],
                key=f"media_filter{key_suffix}",
                help="Select media types to filter the toxicology data based on their measurement units"
            )
        # Define current_chemical_options for database workflow
    chem_df = pd.DataFrame(st.session_state.get('chemicals_data', []))
    if not chem_df.empty and 'chemical_name' in chem_df.columns:
        current_chemical_options = ["-- Select Chemical --"] + chem_df['chemical_name'].dropna().astype(str).str.strip().unique().tolist()
    else:
        current_chemical_options = ["-- No Chemical Names Found --"]
    selected_chemicals = st.multiselect(
        "Select Chemicals from Database",
        options=current_chemical_options,
        key=f"selected_chemicals_supabase",
        help="Hold Ctrl/Cmd or use checkboxes to select multiple chemicals. Start typing to filter."
    )
    # Filter out the placeholder
    selected_chemicals = [c for c in selected_chemicals if c != "-- Select Chemical --"]
    endpoint_type = st.radio(
        "Endpoint Type", ('Acute (LC50, EC50)', 'Chronic (NOEC, LOEC, EC10)'), index=0,
        key=f"endpoint_type{key_suffix}",
        help="Select the general type of endpoint to include."
    )
if uploaded_file is not None:
    min_species = st.number_input(
        "Minimum Number of Species", min_value=3, value=5, step=1,
        key=f"min_species{key_suffix}",
        help="Minimum unique species required after filtering."
    )
    required_taxa_broad = st.multiselect(
        "Required Taxonomic Groups", options=list(TAXONOMIC_MAPPING.keys()), default=list(TAXONOMIC_MAPPING.keys())[:3],
        key=f"required_taxa_broad{key_suffix}",
        help="Select the broad taxonomic groups that *must* be represented."
    )
    data_handling = st.radio(
        "Handle Multiple Values per Species", ('Use Geometric Mean', 'Use Most Sensitive (Minimum Value)'), index=0,
        key=f"data_handling{key_suffix}",
        help="How to aggregate multiple data points for the same species."
    )
else:
        # --- Main instructions at the top of the main page ---
        st.markdown("## Upload a file or fetch chemicals from the database to begin.")
        st.markdown(
            "Upload a file using the sidebar to populate the chemical list and enable analysis."
        )
        st.markdown("""
Upload your **processed** ecotoxicity data file (e.g., a `.csv` containing the required columns:
`test_cas`, `chemical_name`, `species_scientific_name`, `species_common_name`, `species_group`,
`endpoint`, `effect`, `conc1_mean`, `conc1_unit`). Select the chemical from the dropdown and
configure parameters to generate the SSD.
""")
        # Always show these options so distribution_fit and hcp_percentile are defined
        distribution_fit = st.selectbox(
            "Distribution for Fitting", ('Log-Normal', 'Log-Logistic'), index=0,
            help="Statistical distribution to fit to the log-transformed data."
        )
        hcp_percentile = st.number_input(
            "Hazard Concentration (HCp) Percentile", min_value=0.1, max_value=99.9, value=5.0, step=0.1, format="%.1f",
            help="The percentile 'p' for which to calculate the HCp (e.g., 5 for HC5)."
        )

    # *** MODIFIED: Button enabling logic for multi-select ***
        is_ready_to_generate = (
            uploaded_file is not None and
            selected_chemicals and
            all([c not in (None, "-- Upload File First --", "-- Error Reading File --") for c in selected_chemicals])
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
        st.write(f"1. Filtering for Chemicals: {', '.join(selected_chemicals)}")
        # Ensure column is string type and strip whitespace just in case
        df[name_col] = df[name_col].astype(str).str.strip()
        # Filter for selected chemicals (list)
        chem_filter = df[name_col].isin(selected_chemicals)
        filtered_df = df[chem_filter].copy()
        # --- End Modification ---

        if filtered_df.empty:
            st.warning(f"⚠️ No data found for selected chemicals: {', '.join(selected_chemicals)}.")
            st.stop()

        st.write(f"   Found {len(filtered_df)} initial records for the selected chemical(s).")

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
                agg_value = np.exp(np.log(positive_values).mean()) if not positive_values.empty else np.nan
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
        # Map UI label to internal value for SSD distribution
        dist_map = {
            "Log-Normal": "lognormal",
            "Log-Logistic": "loglogistic"
        }
        ssd_dist = dist_map.get(distribution_fit, "lognormal")
        hcp_value, fit_params, plot_data_dict, error_msg = calculate_ssd(
            data=species_df, species_col=species_col, value_col='aggregated_value',
            dist_name=ssd_dist, p_value=hcp_percentile / 100
        )
        if error_msg: st.error(f"❌ Error during SSD calculation: {error_msg}"); st.stop()
        st.success("   ✓ SSD calculation successful.")

        # --- Display Results --- (Keep as is)
        with results_area:
            st.subheader("📊 Results")
            if hcp_value is None:
                st.warning("SSD value (hcp_value) is missing.")
            else:
                st.metric(label=f"Hazard Concentration HC{hcp_percentile}", value=f"{hcp_value:.4g} {data_unit}")

        with plot_area:
            st.subheader("📈 SSD Plot")
            if plot_data_dict is None:
                st.warning("SSD plot data is missing.")
            else:
                try:
                    ssd_fig = create_ssd_plot(plot_data_dict, hcp_value, hcp_percentile, distribution_fit, data_unit)
                    st.plotly_chart(ssd_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error plotting SSD: {e}")
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
    pass