# Resume ATS Checker

A Python-based tool that analyzes resumes against job descriptions using Natural Language Processing (NLP) to help optimize resume content for Applicant Tracking Systems (ATS). The tool supports multiple industries including general technical roles, accounting/finance positions, and digital marketing careers.

## Features

- PDF text extraction and processing
- Industry-specific skill pattern matching:
  - Technical skills (Python, Java, AWS, etc.)
  - Accounting skills (GAAP, QuickBooks, Financial Analysis, etc.)
  - Digital Marketing skills (SEO, Social Media Marketing, Google Analytics, etc.)
- Intelligent similarity scoring using:
  - Skills-based matching (70% weight)
  - Overall text similarity (30% weight)
- Detailed analysis reports including:
  - Overall match score
  - Matching skills identification
  - Missing skills detection
  - Keyword analysis
  - Specific improvement suggestions

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download the required spaCy model:
```bash
python -m spacy download en_core_web_lg
# or for smaller models:
# python -m spacy download en_core_web_md
# python -m spacy download en_core_web_sm
```

## Usage

Choose the appropriate checker based on your industry:

```python
# For general technical roles
from resume_ats_checker import ATSChecker

# For accounting/finance positions
from resume_ats_checker_accounts import ATSChecker

# For digital marketing positions
from resume_ats_checker_digital import ATSChecker

# Initialize and use the checker
checker = ATSChecker()
result = checker.analyze_resume("path/to/resume.pdf", job_description)
```

## Analysis Output

The tool provides a comprehensive report including:
- Overall match score (0-100%)
- List of matching skills found in your resume
- Missing skills that appear in the job description
- Relevant keywords found
- Specific suggestions for improvement

## Requirements

- Python 3.7+
- PyPDF2 >= 3.0.0
- spaCy >= 3.5.0
- pandas >= 1.5.0
- scikit-learn >= 1.0.0
- en-core-web-lg >= 3.5.0 (or md/sm variants)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

<!-- ## License

This project is licensed under the MIT License - see the LICENSE file for details. -->

<!-- ## Future Improvements

- Support for more document formats (Word, RTF, etc.)
- Additional industry-specific skill patterns
- GUI interface
- Batch processing capabilities
- Enhanced suggestion system -->
