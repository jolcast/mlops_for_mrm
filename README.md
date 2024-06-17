# Model Risk Management with MLOps

Welcome to the [Model Risk Management with MLOps](https://github.com/jolcast/mlops_for_mrm) repository! This projectaims to illustrate how to use the MLOps framework to address challenges in model risk management. By following this guide, you'll learn best practices for deploying, monitoring, and maintaining machine learning models in a production environment, ensuring robust and reliable performance.

## Table of Contents

- [Introduction](#introduction)
- [Architecture](#architecture)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Introduction

In the modern data-driven world, managing the risks associated with machine 
learning models is critical. This repository demonstrates how MLOps 
practices can be applied to mitigate these risks effectively. MLOps 
combines machine learning, DevOps, and data engineering to streamline the model 
lifecycle from development to deployment and monitoring.

## Architecture

The architecture of this project follows best practices in MLOps and includes the following components:

1. **Data Ingestion:** Collect and preprocess data from various sources.
2. **Model Training:** Train machine learning models using standardized pipelines.
3. **Model Deployment:** Deploy models to production environments using containerization and orchestration tools.
4. **Monitoring and Logging:** Implement monitoring to track model performance and log relevant metrics.

In the Conceptual Information Flow, the typical interactions addressed in 
the exercise are presented, including artifacts, processes, stages 
(environments), and roles. The roles are responsible for executing each 
stage. In the development stage, for instance, the classic interactions 
between processes and outcomes (artifacts in their various versions) are 
illustrated, with the particular challenge of ensuring process traceability 
and understanding the roles' participation. In the production stage, it is 
highlighted that the model undergoes independent testing and monitoring of 
variables of interest, tasks that correspond to other roles. The feedback 
process represents the information that reaches the development team from 
reviews and methodological or technological findings. Although there are 
limitations (resources) in representing all the interactions between roles 
and environments, this flow has been included to highlight the eventual 
complexity in communication.
![Flujo de Datos conceptual.jpg](..%2F..%2F..%2F..%2F..%2FDownloads%2FFlujo%20de%20Datos%20conceptual.jpg)
Specifically, it describes the transition from raw data ingestion to the 
production deployment of the model, including information preprocessing, 
splitting into training and validation datasets, parameter tuning 
(experimentation), validation, and subsequent selection of the best model. 
While providing an exhaustive methodological description for each of these 
stages is not the scope of this document, each will be addressed 
circumstantially in the section on challenges solved in the context of this 
application.
![Flujo de Datos sobre la arquitectura.jpg](..%2F..%2F..%2F..%2F..%2FDownloads%2FFlujo%20de%20Datos%20sobre%20la%20arquitectura.jpg)

## Data
### Column Descriptions
- **uniqueid**: Unique identifier for each vehicle loan. This ID is used to uniquely identify and track each loan in the dataset.
- **ltv**: Loan-to-Value ratio, indicating the ratio of the loan amount to the value of the vehicle. A higher LTV ratio may indicate a higher risk for the lender, as the borrower has financed a larger portion of the vehicle's value.
- **manufacturer_id**: Identifier for the vehicle manufacturer. This helps in categorizing and analyzing loans based on different vehicle manufacturers.
- **employment_type**: Employment status of the borrower, either Self-employed or Salaried. This can impact the borrower's income stability and ability to repay the loan.
- **state_id**: Identifier for the state where the loan was issued. Different states may have varying regulations, economic conditions, and risk factors that can affect loan performance.
- **perform_cns_score**: Credit score of the borrower. A higher credit score generally indicates a lower risk of default, as it reflects the borrower's creditworthiness.
- **ageinmonths**: Age of the borrower in months. This can be used to analyze trends and correlations between the borrower's age and loan performance.
- **dayssincedisbursement**: Number of days since the loan was disbursed. This can help track the loan's age and its performance over time.
- **loan_default**: Indicator of whether the loan defaulted (1) or not (0). This is the target variable for risk analysis and modeling, indicating the outcome of the loan.

### Summary Statistics
- **uniqueid (Integer)**: The dataset contains 233,154 unique loan IDs.
- **ltv (Float)**: The mean Loan-to-Value ratio is 74.75 with a standard 
  deviation of 11.46. The minimum LTV is 10.03, and the maximum is 95.00.
- **manufacturer_id (Integer)**: The mean manufacturer ID is 69.03 with a 
  standard deviation of 22.14. The values range from 45 to 156.
- **employment_type (String)**: There are 225,493 entries for this variable, 
  with two types: Self-employed and Salaried.
- **state_id (Integer)**: The mean state ID is 7.26 with a standard 
  deviation of 4.48. The values range from 1 to 22.
- **perform_cns_score (Float)**: The mean credit score is 650.90 with a 
  standard deviation of 153.42. The scores range from 300 to 890.
- **ageinmonths (Float)**: The mean age of borrowers is 416.44 months 
  (approximately 34.7 years) with a standard deviation of 117.93 months.
- **dayssincedisbursement (Float)**: The mean number of days since loan 
  disbursement is 99.58 days with a standard deviation of 27.35. The range is from 62 to 153 days.
- **loan_default (Integer)**: 21.71% of the loans have defaulted.

### Missing Values
- **employment_type**: 7,661 missing values
- **perform_cns_score**: 129,785 missing values

The dataset has a significant amount of missing values for 
`perform_cns_score` and some missing values for `employment_type`. These 
will need to be addressed in further analysis or model preparation.  As 
this code version is built, the missing value is handled as a category by 
the xgboost mode.

## Installation

To get started with this project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/jolcast/mlops_for_mrm.git
    cd mlops_for_mrm
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Set up environment variables (if applicable).

## Usage

This section provides a step-by-step guide on how to use the various components of the project.

### Training a Model

1. Preprocess the data:
    ```sh
    python data/preprocessdata.py
    python data/trainvalsplit.py
    ```

2. Train the model:
    ```sh
    python modelling/hyperparametersearch.py
    python modelling/train.py
    python modelling/test.py
    ```

### Deploying a Model

1. Build the Docker image:
    ```sh
    docker build -t yourusername/model-deployment .
    ```

2. Deploy the container:
    ```sh
    docker run -d -p 5000:5000 yourusername/model-deployment
    ```


## Contributing

We welcome contributions to this project! If you have suggestions or improvements, please create a pull request or open an issue.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.


---

Thank you for your interest in Model Risk Management with MLOps. I hope 
this repository helps you effectively manage model risks in your machine
learning projects. If you have any questions or need further assistance, 
feel free to contact me or open an issue.
```