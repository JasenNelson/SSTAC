import streamlit as st 
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO
import supabase
from st_supabase_connection import SupabaseConnection
import plotly.express as px
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def show_media_distribution():
    if st.session_state.chemicals_loaded and st.session_state.chemicals_data:
        with st.expander("Media Distribution", expanded=False):
            try:
                # Create DataFrame from chemicals_data
                df = pd.DataFrame(st.session_state.chemicals_data)
                
                # Ensure required columns exist
                if 'media' not in df.columns:
                    st.error("No media classification data available")
                    st.warning("Please fetch chemical data first to generate the visualization.")
                    return
                
                # Get media from units
                media_counts = df['media'].value_counts()
                fig = px.pie(values=media_counts.values, names=media_counts.index,
                            title="Chemical Distribution by Media",
                            color_discrete_sequence=px.colors.qualitative.Plotly)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating media distribution visualization: {str(e)}")
                st.exception(e)

# Set page configuration first
st.set_page_config(layout="wide")

def initialize_supabase_connection():
    """Initialize and test Supabase connection with proper error handling.
    Returns:
        SupabaseConnection: The initialized Supabase connection or None if failed
    """
    try:
        # First try to get credentials from Streamlit secrets (for Streamlit Cloud)
        supabase_config = st.secrets.get("supabase", {})
        
        # Get URL and key from config
        supabase_url = supabase_config.get("url")
        supabase_key = supabase_config.get("anon_key")
        
        # If not found in secrets, try environment variables (for local development)
        if not supabase_url or not supabase_key:
            # Try environment variables
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
                st.error("   - Add the following variables:")
                st.error("     - SUPABASE_URL=your-project-url")
                st.error("     - SUPABASE_KEY=your-anon-key")
                return None
        
        # Create Supabase connection
        supabase_conn = st.connection(
            "supabase",
            type=SupabaseConnection,
            url=supabase_url,
            key=supabase_key
        )
        
        # Test the connection
        try:
            # Try a simple query to verify basic connection
            # Use a query that doesn't require auth
            test_query = supabase_conn.table("public.chemicals").select("id").limit(1).execute()
            if test_query.data:
                st.success("Successfully connected to Supabase!")
                st.write("Supabase connection test succeeded.")
            else:
                st.warning("Connected to Supabase, but no chemicals found.")
                st.warning("This is normal if you haven't added any chemicals yet.")
                
        except Exception as e:
            st.error(f"Error testing Supabase connection: {str(e)}")
            st.error("Please check:")
            st.error("1. Your Supabase URL and anon key are correct")
            st.error("2. The URL is accessible")
            st.error("3. The anon key has the correct permissions")
            st.error("4. The database table 'public.chemicals' exists")
            st.error("5. RLS policies are configured correctly")
            st.error("""To create the chemicals table, run this SQL:
CREATE TABLE public.chemicals (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    cas_number VARCHAR(20),
    group VARCHAR(100),
    occurrences INTEGER DEFAULT 1
);
""")
            st.error("After creating the table, you'll need to:")
            st.error("1. Add RLS policies to allow read access")
            st.error("2. Insert some initial data")
            st.error("3. Make sure the anon key has proper permissions")
            return None
                
    except Exception as e:
        # Handle other initialization errors
        st.error(f"Failed to initialize Supabase connection: {str(e)}")
        st.error("Full traceback:")
        st.error(traceback.format_exc())
        return None
    
    return supabase_conn

# Initialize session state
if 'chemicals_loaded' not in st.session_state:
    st.session_state.chemicals_loaded = False
if 'chemicals_data' not in st.session_state:
    st.session_state.chemicals_data = []

# Test the connection
try:
    # First try a simple query to test basic connection
    st.info("Testing connection...")
    try:
        # Try to fetch a list of tables to verify connection
        tables_query = supabase_conn.rpc("get_tables").execute()
        if tables_query.data:
            st.success("Successfully connected to Supabase!")
            st.write("Available tables:", tables_query.data)
            
            # Now check if chemicals table exists
            try:
                chemicals_query = supabase_conn.table("chemicals").select("name").limit(1).execute()
                if chemicals_query.data:
                    st.success("Chemicals table found and accessible!")
                else:
                    st.warning("Connected to Supabase, but no chemicals found.")
            except Exception as e:
                if isinstance(e, dict) and e.get('code') == '42P01':
                    st.error("Database table 'chemicals' not found")
                    st.error("Please create the chemicals table in your Supabase database with the following structure:")
                    st.code("""
CREATE TABLE public.chemicals (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    cas_number VARCHAR(20),
    group VARCHAR(100),
    occurrences INTEGER DEFAULT 1
);
""")
                    st.error("After creating the table, you'll need to:")
                    st.error("1. Add RLS policies to allow read access")
                    st.error("2. Insert some initial data")
                    st.error("3. Make sure the anon key has proper permissions")
                else:
                    raise e
        else:
            st.error("Unable to access any tables in the database")
            st.error("Please check:")
            st.error("1. The anon key has the correct permissions")
            st.error("2. RLS policies are correctly configured")
            raise Exception("No tables accessible with current credentials")
            
    except Exception as e:
        # If the RPC call fails, try a simpler query
        st.info("Testing connection with simpler query...")
        try:
            # Try a simpler query to verify basic connection
            test_query = supabase_conn.table("auth.users").select("id").limit(1).execute()
            if test_query.data:
                st.success("Successfully connected to Supabase!")
                st.warning("Note: The 'auth.users' table exists, but we need the 'chemicals' table for this app.")
                
                # Check chemicals table again
                try:
                    chemicals_query = supabase_conn.table("chemicals").select("name").limit(1).execute()
                    if chemicals_query.data:
                        st.success("Chemicals table found and accessible!")
                    else:
                        st.warning("Connected to Supabase, but no chemicals found.")
                except Exception as e:
                    if isinstance(e, dict) and e.get('code') == '42P01':
                        st.error("Database table 'chemicals' not found")
                        st.error("Please create the chemicals table in your Supabase database with the following structure:")
                        st.code("""
CREATE TABLE public.chemicals (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    cas_number VARCHAR(20),
    group VARCHAR(100),
    occurrences INTEGER DEFAULT 1
);
""")
                        st.error("After creating the table, you'll need to:")
                        st.error("1. Add RLS policies to allow read access")
                        st.error("2. Insert some initial data")
                        st.error("3. Make sure the anon key has proper permissions")
                    else:
                        raise e
            else:
                st.error("Unable to access any tables in the database")
                st.error("Please check:")
                st.error("1. The anon key has the correct permissions")
                st.error("2. RLS policies are correctly configured")
                raise Exception("No tables accessible with current credentials")
        except Exception as e:
            st.error(f"Error testing connection: {str(e)}")
            raise e
except Exception as e:
    st.error(f"Connection test failed: {str(e)}")
    st.error("Please check:")
    st.error("1. Your Supabase URL and anon key are correct")
    st.error("2. The URL is accessible")
    st.error("3. The anon key has the correct permissions")
    st.error("4. The database table 'chemicals' exists")
    st.error("Full traceback:")
    import traceback
    st.error(traceback.format_exc())
    raise e
                
    import traceback
    st.error(traceback.format_exc())

supabase_conn = None

# Initialize session state
if 'chemicals_loaded' not in st.session_state:
    st.session_state.chemicals_loaded = False
    st.session_state.chemicals_data = []

# Initialize Supabase connection
supabase_conn = initialize_supabase_connection()

# If connection failed, don't proceed with the rest of the app
if supabase_conn is None:
    st.stop()
    st.warning("Supabase connection failed. Some features may not be available.")
    raise SystemExit("Supabase connection not available")

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
    try:
        # Calculate parameters using numpy
        if dist_name == 'lognormal':
            # For lognormal, we need to transform data
            log_data = np.log(data[value_col])
            mean = np.mean(log_data)
            std = np.std(log_data)
            median = np.exp(mean)
            # Calculate HCP using inverse CDF
            hcp = np.exp(mean + std * np.sqrt(2) * np.sqrt(-2 * np.log(1 - p_value)))
            params = (mean, std)
        elif dist_name == 'normal':
            mean = np.mean(data[value_col])
            std = np.std(data[value_col])
            median = mean
            # Calculate HCP using inverse CDF
            hcp = mean + std * np.sqrt(2) * np.sqrt(-2 * np.log(1 - p_value))
            params = (mean, std)
        else:  # weibull
            # For Weibull, we'll use a simple approximation
            mean = np.mean(data[value_col])
            std = np.std(data[value_col])
            median = mean
            # Calculate HCP using inverse CDF approximation
            hcp = mean + std * np.sqrt(2) * np.sqrt(-2 * np.log(1 - p_value))
            params = (mean, std)
        
        return {
            'median': median,
            'hcp': hcp,
            'params': params,
            'distribution': dist_name
        }
    except Exception as e:
        st.error(f"Error calculating SSD: {str(e)}")
        return None
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
st.title("🧪 Species Sensitivity Distribution (SSD) Generator")

# Using pre-initialized Supabase connection from initialization section
if supabase_conn:
    with st.expander("Chemical Management", expanded=False):
        st.write("Manage your chemical database:")
        
        # Add fetch chemicals button with unique key
        if st.button("Fetch Chemical List from Supabase", key="fetch_chemicals_btn"):
            with st.spinner("Fetching chemical list from Supabase..."):
                fetch_chemicals()

        # Add search box
        search_term = st.text_input("Search Chemicals", key="chem_search")
        
        # Add group filter
        group_options = st.multiselect(
            "Filter by Group",
            options=["All", 
                    'Organic Compounds (Aliphatic)', 'Organic Compounds (Aromatic)',
                    'Inorganic Compounds (Acids)', 'Inorganic Compounds (Bases)', 'Inorganic Compounds (Salts)',
                    'Metals (Heavy Metals)', 'Metals (Transition Metals)', 'Metals (Alkali Metals)',
                    'Pesticides (Herbicides)', 'Pesticides (Insecticides)', 'Pesticides (Fungicides)',
                    'Pharmaceuticals (Antibiotics)', 'Pharmaceuticals (Antivirals)', 'Pharmaceuticals (Analgesics)',
                    'Plastics (Thermoplastics)', 'Plastics (Thermosets)', 'Plastics (Biodegradable)',
                    'Solvents (Polar)', 'Solvents (Non-polar)', 'Solvents (Aprotic)',
                    'Surfactants (Anionic)', 'Surfactants (Cationic)', 'Surfactants (Non-ionic)',
                    'Dyes (Acidic)', 'Dyes (Basic)', 'Dyes (Direct)',
                    'Industrial Chemicals (Corrosives)', 'Industrial Chemicals (Flammables)', 'Industrial Chemicals (Toxic)',
                    'Nanomaterials (Metallic)', 'Nanomaterials (Carbon-based)', 'Nanomaterials (Oxide)',
                    'Radioactive Substances (Alpha)', 'Radioactive Substances (Beta)', 'Radioactive Substances (Gamma)',
                    'Biocides (Disinfectants)', 'Biocides (Antimicrobials)', 'Biocides (Preservatives)',
                    'Food Additives (Preservatives)', 'Food Additives (Colorants)', 'Food Additives (Emulsifiers)',
                    'Cosmetics (Skin Care)', 'Cosmetics (Hair Care)', 'Cosmetics (Makeup)'],
            default=["All"],
            key="group_filter",
            help="Select chemical groups to filter the search results"
        )

        # Add media filter
        media_options = st.multiselect(
            "Filter by Media",
            options=['All', 'Water/Wastewater', 'Soil/Sediment', 'Air', 'Biota', 'Food'],
            default=['All'],
            key="media_filter",
            help="Select media types to filter the chemicals based on their measurement units"
        )

# Show search results
if st.session_state.chemicals_loaded:
            try:
                # Fetch chemicals from database
                chemicals = supabase_conn.table("chemicals").select("name, cas_number").execute()
                
                # Convert to DataFrame and add unique identifier
                if chemicals:
                    df = pd.DataFrame(chemicals.data)
                else:
                    df = pd.DataFrame()
                df['id'] = df.index + 1
                
                # Add chemical group column
                df['group'] = df['name'].apply(get_chemical_group)
                
                # Add CAS number column (if not present)
                if 'cas_number' not in df.columns:
                    df['cas_number'] = ""
                
                # Add media classification based on units
                df['media'] = df['cas_number'].apply(lambda x: get_media_from_unit(x) if x else 'Unknown')
                
                # Add occurrence count
                df['occurrences'] = 1
                
                # Store in session state
                st.session_state.chemicals_data = df.to_dict('records')
                st.session_state.chemicals_loaded = True
                
                # Show success message
                st.success("Successfully fetched chemicals!")
                
            except Exception as e:
                st.error(f"Failed to fetch chemicals: {str(e)}")
                st.exception(e)

def show_media_distribution():
    if st.session_state.chemicals_loaded and st.session_state.chemicals_data:
        with st.expander("Media Distribution", expanded=False):
            try:
                # Create DataFrame from chemicals_data
                df = pd.DataFrame(st.session_state.chemicals_data)
                
                # Ensure required columns exist
                if 'media' not in df.columns:
                    st.error("No media classification data available")
                    st.warning("Please fetch chemical data first to generate the visualization.")
                    return
                
                # Get media from units
                media_counts = df['media'].value_counts()
                fig = px.pie(values=media_counts.values, names=media_counts.index,
                            title="Chemical Distribution by Media",
                            color_discrete_sequence=px.colors.qualitative.Plotly)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating media distribution visualization: {str(e)}")
                st.exception(e)

def fetch_chemicals():
    with st.spinner("Fetching chemical list from Supabase..."):
        try:
            # Fetch chemicals from database
            chemicals = supabase_conn.table("chemicals").select("name, cas_number").execute()
            
            # Check if we got any data
            if not chemicals.data:
                st.error("No chemicals found in the database")
                return
                
            # Convert to DataFrame and add unique identifier
            df = pd.DataFrame(chemicals.data)
            df['id'] = df.index + 1
            
            # Add chemical group column
            df['group'] = df['name'].apply(get_chemical_group)
            
            # Add CAS number column if not present
            if 'cas_number' not in df.columns:
                df['cas_number'] = ""
            
            # Add media classification based on units
            df['media'] = df['cas_number'].apply(lambda x: get_media_from_unit(x) if x else 'Unknown')
            
            # Process chemicals
            chem_data = []
            seen_chemicals = set()
            chemical_groups = {}
            
            for chem in df.itertuples(index=False):
                if chem.name and chem.name not in seen_chemicals:
                    # Determine chemical group based on species group
                    species_group = chem.group if chem.group else "Unknown"
                    chemical_group = get_chemical_group(species_group)
                    
                    # Track chemical groups
                    if chemical_group not in chemical_groups:
                        chemical_groups[chemical_group] = 0
                    chemical_groups[chemical_group] += 1
                    
                    # Add chemical data
                    chem_data.append({
                        'id': chem.id,
                        'name': chem.name,
                        'cas_number': chem.cas_number,
                        'group': chemical_group,
                        'occurrences': 1
                    })
                    seen_chemicals.add(chem.name)
                elif chem.name in seen_chemicals:
                    # Update count for existing chemicals
                    for item in chem_data:
                        if item['name'] == chem.name:
                            item['occurrences'] += 1
                            break
            
            # Store in session state
            st.session_state.chemicals_data = chem_data
            st.session_state.chemicals_loaded = True
            
            # Update chemical options
            if 'file_processed_chem_list' not in st.session_state:
                st.session_state.file_processed_chem_list = []
            st.session_state.file_processed_chem_list = [chem['name'] for chem in chem_data]
            
            st.success(f"Successfully fetched {len(chem_data)} unique chemicals from Supabase!")
            
            # Show chemical group distribution
            st.write("Chemical Group Distribution:")
            group_df = pd.DataFrame(list(chemical_groups.items()), columns=['Group', 'Count'])
            st.bar_chart(group_df.set_index('Group'))
            
            # Show chemical details
            if st.checkbox("Show chemical details", key="show_chem_details"):
                st.write("Chemical Details:")
                chem_df = pd.DataFrame(chem_data)
                st.dataframe(chem_df, hide_index=True)
                
        except Exception as e:
            st.error(f"Failed to fetch chemicals: {e}")
            st.write("Error details:")
            st.write(f"Type: {type(e).__name__}")
            st.write(f"Message: {str(e)}")
            st.exception(e)  # Show full traceback
            raise e  # Re-raise to terminate the app if needed

# Show search results
if st.session_state.chemicals_loaded:
    # Filter by search term
    filtered_chems = st.session_state.chemicals_data
    if search_term:
        filtered_chems = [chem for chem in filtered_chems 
                        if search_term.lower() in chem['name'].lower()]
    
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
            options=chem_df['name'].tolist(),
            help="Select multiple chemicals by holding Ctrl/Cmd"
        )
        
        if selected_chemicals:
            # Show selected chemicals
            selected_df = chem_df[chem_df['name'].isin(selected_chemicals)]
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