# Species Sensitivity Distribution (SSD) Generator

A Streamlit-based web application for generating Species Sensitivity Distribution (SSD) plots following the CCME Protocol for the Protection of Aquatic Life.

## Features

- Generate SSDs for multiple chemicals
- Upload your own data or use the built-in database
- Supports multiple distribution types (Log-Normal, Log-Logistic, Weibull)
- Interactive plots with Plotly
- Data filtering and processing according to CCME guidelines
- Detailed source information and references

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/JasenNelson/SSTAC.git
   cd SSTAC
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   streamlit run ssd_app_v2.py
   ```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Select your data source (upload a file or use the built-in database)

4. Choose the chemicals and parameters for your analysis

5. Click "Generate SSD" to create the plot

## Data Format

When uploading your own data, ensure it follows this format:

- `chemical_name`: Name of the chemical
- `species_scientific_name`: Scientific name of the species
- `broad_group`: Taxonomic group
- `conc1_mean`: Mean concentration value
- `endpoint`: Type of endpoint measurement
- Additional metadata columns as needed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback, please open an issue on GitHub.
