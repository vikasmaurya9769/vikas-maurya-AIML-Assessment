# EDA Analysis

## Dataset Overview

### Leads Dataset

* Total Records: 2,045
* Unique Leads: 2,025
* Features: 21

### Interactions Dataset

* Total Records: 40,000
* Features: 36

Since the dataset did not contain a predefined `converted` column, the target variable was derived using high-intent actions such as demo requests, free-trial starts, contact form submissions, and completed forms.

---

## Data Quality Analysis

### Missing Values

A small number of missing values were found in several categorical columns:

| Column              | Missing Values |
| ------------------- | -------------- |
| Browser             | 101            |
| Company Size        | 101            |
| Annual Revenue Band | 42             |
| City                | 40             |

These values were imputed using an **"Unknown"** category.

### Duplicate Records

The leads dataset contained **20 duplicate lead IDs**, resulting in 2,045 records but only 2,025 unique leads. Duplicate records were removed during preprocessing to maintain one record per lead.

### Outlier Analysis

The `employee_count` feature showed a highly skewed distribution.

* Median Employee Count: 348
* Maximum Employee Count: 250,000

These large values appear to represent legitimate enterprise organizations rather than data-entry errors, so they were retained for further analysis.

---

## Key Findings

### 1. Lead Source Analysis

Lead quality varied significantly across acquisition channels.

| Source         | Conversion Rate |
| -------------- | --------------- |
| LinkedIn       | ~37%            |
| Google         | ~37%            |
| Email Campaign | ~36%            |
| Instagram      | ~15%            |

LinkedIn and Google generated the highest-quality leads, while Instagram produced substantially lower conversion rates.

---

### 2. Lead Segment Analysis

| Segment    | Conversion Rate |
| ---------- | --------------- |
| Enterprise | ~39%            |
| Mid-Market | ~33%            |
| Startup    | ~23%            |
| SMB        | ~19%            |

Enterprise leads converted at roughly twice the rate of SMB leads, indicating that larger organizations may have stronger purchasing intent and budget availability.

---

### 3. Funnel Progression Analysis

Funnel depth emerged as one of the strongest indicators of conversion.

| Maximum Funnel Stage Reached | Conversion Rate |
| ---------------------------- | --------------- |
| Awareness                    | ~2%             |
| Consideration                | ~15%            |
| Evaluation                   | ~34%            |
| Decision                     | ~63%            |

Leads that progressed to the Decision stage were significantly more likely to convert than those who remained at the top of the funnel.

---

### 4. Behavioral Engagement Analysis

Converted leads showed noticeably stronger engagement than non-converted leads.

| Metric             | Converted | Non-Converted |
| ------------------ | --------- | ------------- |
| Average Sessions   | 8.4       | 3.6           |
| Pricing Page Views | 2.9       | 1.3           |

This suggests that repeat engagement and pricing-page activity are strong indicators of purchase intent.

---

### 5. Industry and Company Characteristics

Industry, funding stage, company size, and job role all showed measurable differences in conversion behavior.

Technology-focused and more mature organizations generally demonstrated stronger conversion performance than smaller or early-stage companies.

These characteristics are expected to provide useful predictive signals during model training.

---

## Correlation Analysis

Correlation analysis showed positive relationships between engagement-related features and conversion behavior.

No severe multicollinearity issues were observed among the primary engineered features.

---

## Supporting Visualizations

The following visualizations were generated during exploratory analysis and are available in the `outputs/` directory:

* Conversion Distribution
* Conversion by Source
* Conversion by Segment
* Conversion by Industry
* Conversion by Funding Stage
* Funnel Conversion Analysis
* Session Behavior Analysis
* Employee Count Outlier Analysis
* Correlation Heatmap

---

## Conclusion

The exploratory analysis indicates that both lead attributes and behavioral engagement contribute significantly to conversion likelihood.

Among all factors examined, funnel progression, session activity, pricing-page engagement, and repeat visits emerged as the strongest indicators of conversion.

These findings were used to guide feature engineering and model development in the next stage of the project.
