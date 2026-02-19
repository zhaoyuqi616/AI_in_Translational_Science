# 🌐 Project 6 — Web Scraping for Cancer Omics Data and Biomedical Metadata Retrieval

---

## 📌 Overview

This project demonstrates automated retrieval of biomedical metadata and cancer-related omics information using web scraping and API-based data extraction techniques.

The workflow includes:

- Scraping TCGA tissue source site codes
- Extracting tabular metadata using BeautifulSoup and pandas
- Accessing UniProt protein records programmatically
- Retrieving protein information via RESTful APIs
- Parsing structured JSON responses
- Collecting PDB, GO, and organism annotations

This project highlights practical data engineering techniques essential for translational bioinformatics and AI-driven omics research.

---

## 🎯 Objective

Automate the extraction of structured biomedical data from public web resources and APIs to support:

- Cancer genomics workflows
- Protein annotation pipelines
- Omics data harmonization
- Metadata-driven machine learning pipelines

---

## 🧪 Data Sources

### 1️⃣ TCGA Tissue Source Site Codes
URL:
https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tissue-source-site-codes

### 2️⃣ cBioPortal
https://www.cbioportal.org

### 3️⃣ UniProt Upload List Service
https://www.uniprot.org/uploadlists/

### 4️⃣ EBI Proteins REST API
https://www.ebi.ac.uk/proteins/api/

---

## 🧠 Case Descriptions

### Case 1 — Web Scraping Using BeautifulSoup

- Retrieve TCGA tissue source site code table
- Parse HTML table
- Convert to pandas DataFrame
- Export as `.txt` file

Output:
Tissue_Source_Site_Codes.txt


---

### Case 2 — HTML Table Extraction Using pandas

- Use `pd.read_html()` to automatically extract tables
- Save results to Excel format

Output:
Tissue_Source_Site_Codes_from_pandas.xls


---

### Case 3 — Basic HTML Parsing

- Retrieve webpage content from cBioPortal
- Extract document title using BeautifulSoup

Demonstrates:
- Basic HTML parsing
- Metadata extraction

---

### Case 4 — UniProt Programmatic Query

Function:
get_uniprot(query='', query_type='ACC')


Features extracted:

- Organism
- Protein size
- PDB structures + resolution
- Gene Ontology (GO) functions
- GO biological processes

Example queried entries:
- P53_HUMAN
- P53_MOUSE
- P53_RAT

Output:
Structured pandas DataFrame containing protein annotations.

---

### Case 5 — RESTful API Access (JSON Parsing)

- Query EBI Proteins API
- Retrieve JSON response
- Parse structured protein data
- Print full JSON object

Demonstrates:
- API authentication
- JSON parsing
- Error handling
- Structured data extraction

---

## 🛠 Implementation Details

### Python Libraries Used

- requests
- BeautifulSoup (bs4)
- pandas
- urllib
- json
- re
- sys

### Key Techniques Demonstrated

- HTML parsing
- Table extraction
- API requests
- JSON handling
- URL encoding
- Text parsing using regex
- DataFrame construction
- File export (CSV, Excel)

---

## 📁 Project Structure


---

## ▶️ How to Run

### 1️⃣ Install Dependencies

```bash
pip install requests beautifulsoup4 pandas lxml
python Machine_Learning_Project_6_-_Web_Scraping_for_Cancer_Omics_Data.py

