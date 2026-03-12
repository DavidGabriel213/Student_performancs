# 🎓 Nigerian Student Academic Performance Analysis

A comprehensive data cleaning, feature engineering and EDA project using Python.

---

## 📋 Project Overview

This project involved cleaning and analyzing a Nigerian student academic dataset
containing 525 student records across 10 universities, 10 courses and 14 states.
Every column had serious quality issues — the goal was to clean everything from scratch,
engineer meaningful academic features and answer real EDA questions through
statistical analysis and visualization.

---

## 🛠️ Tools & Libraries

- **Python**
- **Pandas** — data cleaning & analysis
- **NumPy** — numerical operations, IQR & manual Pearson correlation
- **Matplotlib** — data visualization

---

## 🧹 Data Cleaning Steps

| Column | Problem | Solution |
|---|---|---|
| CGPA | '3.5/5.0' strings, negatives, outliers > 5 | Strip '/5.0', cap at 5, course group fill |
| YearOfStudy | '3L' strings, outliers x100 | Strip 'L', take first digit if len > 1 |
| AttendanceRate | '75%' strings, outliers x10 | Strip '%', divide by 10 if > 100 |
| DailyStudyHours | '5 hrs', negatives, outliers | Strip 'hrs', IQR both bounds |
| ExamScore/Assignment | '75%', negatives, outliers | Strip '%', divide by 10 if > 100 |
| University/Course | Abbreviations: 'unilag', 'CS', 'MED' | Dictionary mapping to full names |
| HasScholarship/Passed | Mixed '1'/'0'/yes/no/NaN | Dictionary mapping + feature-based fill |
| Age | Outliers on both sides | IQR upper AND lower bound detection |
| Duplicates | 25 duplicate rows | drop_duplicates() |

---

## 📐 Feature Engineering

| Feature | Formula |
|---|---|
| Performance Index | (ExamScore x 0.5) + (Assignment x 0.3) + (Attendance x 0.2) |
| Study Efficiency | CGPA / DailyStudyHours |
| Class | CGPA >= 4.5 → First Class, >= 4.0 → 2nd Upper, >= 3.5 → 2nd Lower, >= 2.5 → Third Class |

---

## 🔍 EDA Questions Answered

| Question | Finding |
|---|---|
| Does internet access improve CGPA? | Yes — avg CGPA with internet: 3.06 vs without: 2.98 |
| Do scholarship students study more? | Yes — 6.7 hrs/day vs 6.4 hrs for non-scholarship |
| Does distance affect attendance? | Pearson correlation = 0.078 — virtually no effect |

> 📌 Pearson correlation was computed **manually** using NumPy covariance matrix — not just `.corr()`

---

## 📊 Key Findings

- 🌐 Internet access has a small but positive effect on academic performance
- 🏆 Scholarship students show slightly higher study commitment
- 📏 Distance from campus does not significantly affect attendance
- 🤖 This dataset is **ML-ready** — PerformanceIndex and Class can serve as target variables for supervised learning models

---

## 📁 Files

| File | Description |
|---|---|
| `Code.py` | Full Python source code |
| `student_academic_messy.csv` | Raw messy dataset |

---

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib
python Code.py
