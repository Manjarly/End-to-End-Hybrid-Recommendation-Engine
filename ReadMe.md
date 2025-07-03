# **End-to-End Hybrid Movie Recommendation Engine**

This project implements a hybrid movie recommender system that combines Collaborative Filtering and Content-Based Filtering to provide personalized movie suggestions. The system is built using Python and leverages the MovieLens 100k dataset.

## **🌟 Features**

* **Hybrid Approach:** Combines SVD-based Collaborative Filtering and TF-IDF-based Content-Based Filtering for robust and accurate recommendations.  
* **Hyperparameter Tuning:** Automatically determines the optimal number of latent factors for the SVD model to maximize precision and prevent overfitting.  
* **Cold-Start Handling:** Dynamically serves content-based recommendations to new users who have no rating history, ensuring immediate utility.  
* **Performance Benchmarking:** Evaluates the final model against Popularity and Random baselines using Precision@10 and Recall@10 metrics.  
* **Data Visualization:** Includes clear visualizations for model evaluation and hyperparameter analysis using Matplotlib and Seaborn.

## **⚙️ How It Works**

The engine is comprised of three core components:

1. **Content-Based Filtering:** This model recommends items based on their attributes. It calculates the TF-IDF vectors for movie genres and uses cosine similarity to find movies that are similar to a user's known preferences.  
2. **Collaborative Filtering:** This model leverages the "wisdom of the crowd" by analyzing the rating patterns of all users. It uses **Singular Value Decomposition (SVD)** to perform matrix factorization on the user-item ratings matrix, uncovering latent factors that represent user tastes and movie characteristics.  
3. **Hybrid Engine:** The final model intelligently blends the recommendations from the two models above. It prioritizes the personalized suggestions from collaborative filtering while using content-based results to supplement the list and handle new users.

## **📊 Performance & Evaluation**

The collaborative filtering model was tuned to find the optimal number of latent factors (n\_components). The validation curve below shows how Precision@10 changes with n\_components, helping to select the best value that balances model complexity and performance.

#### **Validation Curve for n\_components**

*This graph shows the model's precision peaking at an optimal number of components before performance plateaus, indicating the best trade-off.*  
The final model was benchmarked against two baseline recommenders. The results clearly demonstrate the effectiveness of the SVD-based collaborative filtering approach.

#### **Model Performance Comparison**

*The CF model significantly outperforms both the Random and Popularity baselines in both Precision@10 and Recall@10.*

| Model | Precision@10 | Recall@10 |
| :---- | :---- | :---- |
| **Our CF Model** | **\~0.26** | **\~0.12** |
| Popularity Recommender | \~0.15 | \~0.06 |
| Random Recommender | \~0.01 | \~0.01 |

## **🚀 Getting Started**

### **Prerequisites**

Make sure you have Python 3.x installed. You can install the necessary libraries using pip:  
pip install pandas numpy scikit-learn matplotlib seaborn requests

## **Usage**

To run the complete pipeline including model training, evaluation, and demonstration, open the ipynb file in Google Collab or VS Code and Run in order.

The script will automatically download the dataset, tune the hyperparameters, train the models, display evaluation metrics and graphs, and print out example recommendations.

## **🛠️ Technologies Used**

* **Python**  
* **Pandas & NumPy:** For data manipulation and numerical operations.  
* **Scikit-learn:** For SVD, TF-IDF, and model evaluation.  
* **Matplotlib & Seaborn:** For data visualization.  
* **Jupyter Notebook** (optional, for experimentation).
