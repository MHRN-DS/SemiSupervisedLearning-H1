# Semi-Supervised Learning via Gradient Descent, Block Coordinate Gradient Descent & Coordinate Minimization
## A Theoretical & Empirical Study of Implementation of the Solvers

This project focuses on implementing and comparing first-order optimization methods applied to **semi-supervised learning (SSL)**, using both synthetic datasets and the real-world **Digits dataset**. Various solver strategies are evaluated based on performance metrics such as classification accuracy, loss, and computational time.

> **Grade received: 30 e laude/30**  
> **Contributors**: [@MaxHorn](https://github.com/MHRN-DS), [@MikhailIsakov](https://github.com/Mishlen337), [@LennartBredthauer](https://github.com/Lenny945)  
> **University of Padova, Optimization for Data Science – May 2025**  
> Supervisor: Prof. Dr. Francesco Rinaldi

---

## Overview

This homework investigates several gradient-based optimization methods tailored to solve the sparse logistic regression problem. The methods are implemented from scratch and tested on:

- **Synthetic datasets** with known sparsity patterns
- **Digits dataset** from `scikit-learn`, a multi-class image classification task

Implemented methods include:

- **Gradient Descent** (step sizes: `1/L` and `2/(L + σ)`)
- **Gradient Descent with Momentum** (Heavy Ball / Polyak)
- **Nesterov Accelerated Gradient Descent**
- **Block Coordinate Gradient Descent (BCGD)**
- **Coordinate Minimization**

---

## Project Structure

```
.
├── syn_data.ipynb               # Experiments on synthetic datasets
├── digits.ipynb                 # Experiments on the Digits dataset
├── solvers_digits.py            # All solver implementations
├── SSL_Report.pdf               # Full report submission
├── data/                        # Digits dataset loaded via sklearn
```

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/MHRN-DS/SemiSupervisedLearning-H1.git
   cd SemiSupervisedLearning-H1
   ```

2. Install dependencies:
   ```bash
   pip install numpy scikit-learn matplotlib pandas
   ```

3. Launch the notebooks:
   ```bash
   jupyter notebook syn_data.ipynb
   jupyter notebook digits.ipynb
   ```

---

## Datasets Used

- **Synthetic data** (generated using NumPy)
- [Digits dataset – scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)

---

## Key Features

- Full implementation of five optimization strategies for sparse logistic regression
- Application to both controlled synthetic and real-world image classification data
- Metrics: accuracy, CPU time, and objective function value
- Visual comparison of solver performance across tasks
- Reproducible experiments using clean and modular Python code

---

## Requirements

- Python 3.8+
- `numpy`
- `scikit-learn`
- `matplotlib`
- `pandas`

To install all required libraries:
```bash
pip install -r requirements.txt
```

---

## License

This repository is intended for academic reference only.  
Unauthorized modification or redistribution is not permitted.
