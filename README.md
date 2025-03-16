
# Machine Learning: Administrative data of the Kosovo Tax Administration - ATK

## Project Information
- Institution: University of Pristina "Hasan Prishtina"
- Program: Master's Degree, Computer and Software Engineering
- Subject: Machine Learning      
- Professor: Prof. Dr. Lule Ahmedi and Dr. Sc. Mërgim H. HOTI
  
  <div align="center">
  <img src="images/universiteti.jpg" alt="Project Logo" width="300">
</div>

## Authors

- [Albin Hashani](https://github.com/AlbinHashanii)
- [Arjana Tërnava](https://github.com/ArjanaaTernava)
- [Erza Osmani](https://github.com/erzaosmani)

## Dataset Information

This dataset sourced from [ATK](https://www.atk-ks.org/open-data/) contains a total of 97992 (without preprocessing it) rows and 12 columns. It is centered on the administrative data of the Kosovo Tax Administration, specifically addressing investments with and without VAT for imports as well as domestic investments, along with key temporal and taxpayer information. The dataset comprises three integer columns: `Viti (year), Muaji (month), and Tatimpaguesit (taxpayer identifier)`, three object columns: `Pershkrimi, Statusi, and Komuna` and six float columns detailing various investment figures. This structured information is essential for analyzing fiscal trends and investment patterns in Kosovo.  

## First Phase

The first phase of the project involves the preparation of the machine learning model to further apply the necessary algorithms.
The steps included:

- Data Loading
- Data overview
- Missing values
- Duplicate values
- Data Aggregation and Sampling
- Exploratory Data Analysis (EDA) on Numerical Features - Skewness
- Outlier Detection and Removal
- Dataset Preparation for Modeling
- Handling Class Imbalance

## Development Environment

- Editor: PyCharm
![PyCharm](https://img.shields.io/badge/-PyCharm-00B300?logo=pycharm&logoColor=white&style=for-the-badge)

- Instructions:
    - Download and install PyCharm Editor
    - Select Pure Python for the project type
    - If you don't have an interpreter set up, you can do it in the editor based on instructions.
    - Install the required packages

  - Results of the first phase: 

    - Data Loading
      ```bash
      csv_data = pd.read_csv("atk-investimet-tvsh.csv", thousands=',')

      ```
      - The dataset file `atk-investimet-tvsh.csv` should be located in the directory: `machine_learning_project/src`.
 
    - Printing Data Types - Data overview
      
       ![Project Logo](images/data_types.png)

    - Number of duplicate_rows

      ![Project Logo](images/number_of_rows.png)

    - Aggregation Functions - mean, count, std, sum, median
      
      ![Project Logo](images/aggregated_data_1.png)
      
      ![Project Logo](images/aggregated_data_2.png)
            
    - Missing values - The result indicates there are no missing values in our dataset: 
      
      ![Project Logo](images/missing_types.png)
      
    - Duplicate values
   
      ![Project Logo](images/duplicate_rows.png)

    - Sampling - Randomly samples 10% of the dataset to inspect a smaller subset for analysis.
      
      ![Project Logo](images/sampled_data.png) 

    - Exploratory Data Analysis (EDA) on Numerical Features - Skewness: The code computes skewness to assess the asymmetry of numerical distributions, using histograms with mean and median markers to visually highlight any skew.
      
      ![Project Logo](images/skewness_numerical_cols.png)
      
      ![Project Logo](images/skewness_without_outliers.png) 
   
  - Skewness data before removing outliers: This part includes visualization of the numerical columns without removing outliers and their skewness. Below you can find the images for each column: 
    
    ![Project Logo](images/viti_skewness.png)

    ![Project Logo](images/muaji_skewness.png)

    ![Project Logo](images/tatimpaguesit_skewness.png)
    
    ![Project Logo](images/blerjet_importet_pa_tvsh_skewness.png)
    
    ![Project Logo](images/blerjet_investive_vendore_8_skewness.png)
    
    ![Project Logo](images/blerjet_investive_vendore_18_skewness.png)
    
    ![Project Logo](images/blerjet_me_tvsh_jo_te_zbritshme_skewness.png)

    ![Project Logo](images/importet_investive_18_skewness.png)  

    ![Project Logo](images/importet_investive_8_skewness.png)
    
  - Skewness data after removing outliers: This part includes visualization of the numerical columns after removing outliers and their skewness. The method that was used for removing outliers was `Z-score method`. The results show that the columns `Viti` and `Muaji` have near-zero skewness, indicating almost symmetrical distributions, while other numerical columns exhibit large positive skewness, reflecting heavy right tails. After applying a `Z-score threshold of 2`, the skewness values in these highly skewed columns decrease. However, the data still remains somewhat skewed, which is common in financial or transactional datasets. Below you can find the images for each column: 
    
       ![Project Logo](images/skewness_without_outliers.png)
    
       ![Project Logo](images/viti_wo_outliers.png)
    
       ![Project Logo](images/muaji_wo_outliers.png)

      ![Project Logo](images/tatimpaguesit_wo_outliers.png)

      ![Project Logo](images/blerjet_importet_investive_pa_tvsh_wo_outliers.png)

      ![Project Logo](images/blerjet_investive_vendore_8_wo_outliers.png)

      ![Project Logo](images/blerjet_investive_vendore_18_wo_outliers.png)

      ![Project Logo](images/importet_tvsh_zbritshme_wo_outliers.png) 

      ![Project Logo](images/importet_8_wo_outliers.png)  

      ![Project Logo](images/importet_18_wo_outliers.png) 
  - Dataset Preparation for Modeling and Handling Class Imbalance

      The target column chosen for modeling is `Statusi`, which plays a crucial role in the classification task. The dataset was first preprocessed by separating this target variable from the features, then splitting the data into training and testing sets to ensure an unbiased evaluation of model performance. Recognizing the class imbalance in `Statusi`, the Synthetic Minority Over-sampling Technique (SMOTE) was applied to the training set to generate synthetic examples for underrepresented classes. This approach ensures that the model is trained on a balanced dataset, ultimately enhancing its ability to accurately generalize to new data.

      ![Project Logo](images/smote.png) 

- The new cleaned dataset `Investimet-2024-cleaned-no-outliers` should be located in the directory: `machine_learning_project/src`. This dataset now contains 94519 rows.

### License

[Apache-2.0](http://www.apache.org/licenses/)
