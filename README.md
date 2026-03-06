# Data Breach Analysis Dashboard

Analyzed ~12,000 data breach records (2000–2024) to find patterns in how breaches happen, which industries get hit hardest, and what role human error plays. Built an interactive Streamlit dashboard on top of the analysis.

## 🚀 Live Dashboard
**Try the interactive dashboard:** [https://data-breach-dashboard.streamlit.app]



## Why I made this

I wanted a hands-on data analysis project that covers the full workflow — loading raw data, cleaning it, doing EDA, visualizing findings, and building something interactive out of it. Cybersecurity breaches felt like an interesting domain with enough categorical + numerical mix to practice different types of analysis.

## The dataset

`dataset/raw/DataBreach_dataset.csv` — 12,378 rows, 10 columns, no nulls, no duplicates.

**Columns:**

| Column | What it is |
|--------|-----------|
| Year | 2000–2024 |
| Company Name | 49 unique companies (Amazon, Facebook, Target, etc.) |
| Type of Breach | 7 types — Phishing, Ransomware, DDoS, Malware, Insider Threat, Password Guessing, Physical Breach |
| Records Compromised | How many records were exposed |
| Financial Loss | Dollar amount of damage |
| Impact Level | Low / Medium / High / Critical |
| Industry | Finance, Government, Healthcare, Retail, Technology |
| Human Error Factor | Weak Password Mgmt, Phishing Attack, Insider Threat, Improper Disposal, Unintentional Info Leak |
| Human Error Factor Code | Numeric code 1–5 for the above |
| Mitigation Measures | Access Controls, Data Encryption, Regular Audits, Security Training, 2FA |

During cleaning I also created `Loss Category` (bucketed financial loss) and `Records Category` (bucketed records) for easier grouping.

## What I found

**Big picture:** ~$18.6B total financial loss across all 12,378 breaches. Average loss per breach is about $1.5M.

**Breach types:** Password Guessing is the most frequent (1,800 incidents). Phishing causes the highest average loss (~$1.53M per incident) even though it's not the most common — so it's the costliest attack type.

**Industries:** Finance leads both in number of breaches (2,530) and average loss per breach. All five industries see all seven breach types pretty evenly though, no industry is immune to any particular attack.

**Human error:** Weak Password Management is the top factor (2,537 incidents). The distribution is fairly even across industries, which suggests this is a systemic problem rather than industry-specific.

**Companies:** Amazon tops the list with 434 breaches and ~$641M total loss.

**Mitigation:** Regular Security Training is the most common measure. But all five measures show up across all impact levels — no single measure prevents high-impact breaches on its own. Layered defense is the way to go.

## Project structure

```
data_breach_dashboard/
├── dashboard.py                    # Streamlit dashboard
├── requirements.txt
├── dataset/
│   ├── raw/DataBreach_dataset.csv
│   └── cleaned/data_breaches_cleaned.csv
├── notebooks/
│   ├── data_analysis.ipynb         # Jupyter notebook version
│   └── data_analysis.py            # Script version (generates charts)
└── charts/                         # 22 PNGs from the analysis
```

## Running it

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Generate the static charts** (optional, they're already in `charts/`):
```bash
python notebooks/data_analysis.py
```

**Launch the dashboard:**
```bash
streamlit run dashboard.py
```
Opens at `http://localhost:8501`. The dashboard has sidebar filters for year range, industry, breach type, impact level, human error factor, and mitigation measure — all charts update in real time.

## Built with

- **pandas / numpy** for data wrangling
- **matplotlib / seaborn** for the static charts
- **plotly** for interactive charts (both in the script and dashboard)
- **kaleido** for exporting plotly charts to PNG
- **streamlit** for the dashboard
