# CAD-RADS AI Analysis System

## Project Overview

This project is an AI-based Coronary Artery Disease Reporting and Data System (CAD-RADS) analysis tool. It is primarily used to compare the consistency and accuracy between AI models and human experts in coronary artery disease diagnosis, providing scientific evidence for the clinical application of medical imaging AI.

## Main Features

### 1. AI Response Data Processing (`get_stage.py`)
- **Function**: Extract structured coronary artery analysis information from AI model responses
- **Key Features**:
  - Parse JSON format AI response data
  - Extract segment numbers, stenosis degree, plaque type, and other key information
  - Process myocardial bridge and modifier special markers
  - Support multiple data format input processing

### 2. Statistical Analysis Module (`statistical_analysis.py`)
- **Function**: Comparative analysis of AI and human expert diagnostic results
- **Key Features**:
  - CAD-RADS classification comparison analysis
  - Generate confusion matrices and classification reports
  - Calculate accuracy, sensitivity, specificity, and other metrics
  - Support multiple visualization chart outputs
  - Handle duplicate examination numbers and missing data

### 3. Radar Chart Analysis (`statistical_analysis_radar.py`)
- **Function**: Multi-dimensional analysis visualization based on radar charts
- **Key Features**:
  - Multi-indicator radar chart generation
  - Support AI vs. expert result comparison display
  - Customizable chart styles and color schemes
  - Professional medical chart format output

### 4. Intraclass Correlation Coefficient Calculation (`icc.py`)
- **Function**: Calculate inter-rater reliability
- **Key Features**:
  - Support multiple ICC calculation methods
  - Cohen's Kappa coefficient calculation
  - Handle categorical and continuous variables
  - Evaluate inter-expert diagnostic consistency

### 5. Segment Data Management (`segments.json`)
- **Function**: Store detailed information about coronary artery segments
- **Data Content**:
  - Segment number and name mapping
  - Stenosis classification
  - Plaque type definitions
  - Myocardial bridge marking standards

## Technology Stack

- **Python 3.x**: Primary development language
- **Pandas**: Data processing and analysis
- **NumPy**: Numerical computation
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical charts
- **Scikit-learn**: Machine learning metrics calculation
- **OpenPyXL**: Excel file processing

## Data Flow

```
Raw AI Response → Data Extraction → Structured Processing → Statistical Analysis → Visualization Output
     ↓                ↓                    ↓                      ↓                      ↓
segments.json     get_stage.py        Processing Module      analysis.py           Chart Generation
```

## Usage Guide

### 1. Data Processing
```python
from get_stage import Process

processor = Process()
data = processor.read_data('your_data_file.xlsx')
# Further processing...
```

### 2. Statistical Analysis
```python
from statistical_analysis import StatisticalAnalysis

analyzer = StatisticalAnalysis()
analyzer.statistical_analysis('ai_results.xlsx', 'human_results.xlsx')
```

### 3. ICC Calculation
```python
# Run icc.py directly for intraclass correlation coefficient calculation
python icc.py
```

## Output Results

- **Statistical Reports**: Detailed diagnostic consistency analysis
- **Visualization Charts**: Confusion matrices, ROC curves, radar charts, etc.
- **Consistency Metrics**: Kappa coefficient, ICC, sensitivity, specificity, etc.
- **Excel Reports**: Structured analysis result output

## Data Requirements

### Input Data Format
- **AI Result File**: Excel format, containing examination number, AI diagnostic results, and other fields
- **Expert Result File**: Excel format, containing examination number, expert diagnostic results, and other fields
- **Required Fields**: Examination number, CAD-RADS classification, SIS score, P classification, etc.

### Data Preprocessing
- Automatic handling of duplicate examination numbers
- Missing value filling and cleaning
- Data format standardization

## Important Notes

1. **Data Privacy**: Ensure privacy protection and compliant use of medical data
2. **Result Interpretation**: Statistical results require professional medical background for interpretation
3. **Quality Control**: Regular validation of analysis result accuracy
4. **Version Control**: Maintain consistency between data and code versions

## Project Structure

```
CAD-RADS/
├── get_stage.py              # AI response data processing
├── statistical_analysis.py   # Basic statistical analysis
├── statistical_analysis_radar.py  # Radar chart analysis
├── icc.py                    # Intraclass correlation coefficient calculation
├── segments.json             # Segment data definition
└── README.md                 # Project documentation
```

## Contact Information

This project is used for medical imaging AI research. For technical questions or collaboration requests, please contact the project team.

---

**Disclaimer**: This tool is for research purposes only and should not be used directly for clinical diagnostic decisions. 