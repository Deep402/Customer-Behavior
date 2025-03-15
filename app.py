import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cohere
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# Initialize Cohere AI (Replace with your actual API key)
co = cohere.Client('KTn7ndyWTyFbx9yGwzrS27JYOy0TRjttcObYzk5t')

# ------------------------- Step 1: User Input for Business Type -------------------------
business_type = input("Enter the business type (e.g., E-commerce, Retail, SaaS, Finance, etc.): ").strip()

# ------------------------- Step 2: Sample Dataset -------------------------
data = {
    "customer_id": range(1, 21),
    "total_spent": np.random.randint(500, 150000, size=20),  
    "purchase_frequency": np.random.randint(1, 50, size=20),  
    "last_purchase_days": np.random.randint(1, 180, size=20),  
    "customer_age": np.random.randint(18, 70, size=20),  
    "location": np.random.choice(["Urban", "Suburban", "Rural"], size=20),  
    "avg_basket_size": np.random.randint(1, 10, size=20),  
    "last_review_rating": np.random.randint(1, 6, size=20)  
}

df = pd.DataFrame(data)

# ------------------------- Step 3: Data Analysis -------------------------
def analyze_data(df):
    """Calculate key business metrics."""
    return {
        "avg_spent": df["total_spent"].mean(),
        "avg_frequency": df["purchase_frequency"].mean(),
        "churn_rate": (df["last_purchase_days"] > 90).mean() * 100,
        "high_basket_customers": (df["avg_basket_size"] > 5).sum(),
        "low_rated_customers": (df["last_review_rating"] < 3).sum()
    }

# ------------------------- Step 4: Optimized Customer Segmentation -------------------------
def segment_customers(df):
    """Efficient customer segmentation using MiniBatchKMeans."""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[["total_spent", "purchase_frequency", "avg_basket_size"]])

    kmeans = MiniBatchKMeans(n_clusters=3, random_state=42, n_init=10, batch_size=5)
    df["segment"] = kmeans.fit_predict(scaled_data)

    segment_labels = {
        0: "High-Value Customers",
        1: "Frequent Shoppers",
        2: "At-Risk Customers"
    }

    df["segment_label"] = df["segment"].map(segment_labels)
    return df

# ------------------------- Step 5: Predictive Analytics -------------------------
def predict_spending(df):
    """Predict future spending based on purchase frequency."""
    X = df[["purchase_frequency"]].values
    y = df["total_spent"].values

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    return {
        "model": model,
        "r2": r2_score(y, y_pred),
        "mae": mean_absolute_error(y, y_pred),
        "prediction": model.predict([[20]])[0]
    }

# ------------------------- Step 6: AI-Powered Business Insights -------------------------
def generate_ai_insights(metrics, business_type, question=None):
    """Generate precise AI insights tailored to the business type."""
    prompt = f"""Analyze this {business_type} customer dataset:
    - Average Spending: ₹{metrics['avg_spent']:,.0f}
    - Purchase Frequency: {metrics['avg_frequency']:.1f} times
    - Churn Risk: {metrics['churn_rate']:.1f}%
    - High Basket Size Customers: {metrics['high_basket_customers']}
    - Customers with Low Ratings: {metrics['low_rated_customers']}

    Provide 3 precise data-driven strategies to:
    - Increase customer retention
    - Maximize revenue growth
    - Improve customer satisfaction"""

    if question:
        prompt += f"\n\nUser's Question: {question}\nProvide a detailed and industry-specific response."

    try:
        response = co.generate(
            model='command',
            prompt=prompt,
            max_tokens=400,
            temperature=0.5
        )
        return response.generations[0].text.strip()
    except Exception as e:
        return f"AI Insights temporarily unavailable. Error: {str(e)}"

# ------------------------- Step 7: Dashboard Visualization (6 Graphs) -------------------------
def visualize_dashboard(df, metrics):
    """Generate customer insights dashboard with 6 graphs."""
    plt.figure(figsize=(18, 10))
    plt.suptitle(f"Customer Analytics Dashboard for {business_type}", fontsize=18)

    # 1. Customer Segmentation Pie Chart
    plt.subplot(2, 3, 1)
    df.segment_label.value_counts().plot.pie(autopct='%1.1f%%', colors=['skyblue', 'orange', 'lightgreen'])
    plt.title("Customer Segmentation")

    # 2. Spending vs Purchase Frequency Scatter Plot
    plt.subplot(2, 3, 2)
    sns.scatterplot(x=df["purchase_frequency"], y=df["total_spent"], hue=df["segment_label"], palette="Set2", s=100)
    plt.title("Spending vs Purchase Frequency")
    plt.xlabel("Monthly Purchases")
    plt.ylabel("Total Spent (₹)")

    # 3. Bar Chart for Customer Locations
    plt.subplot(2, 3, 3)
    sns.countplot(x=df["location"], palette="coolwarm")
    plt.title("Customer Location Distribution")

    # 4. Bar Chart for Customer Age Groups
    plt.subplot(2, 3, 4)
    sns.histplot(df["customer_age"], bins=5, kde=True, color="purple")
    plt.title("Customer Age Distribution")

    # 5. Bar Chart for Average Basket Size
    plt.subplot(2, 3, 5)
    sns.histplot(df["avg_basket_size"], bins=5, kde=True, color="green")
    plt.title("Average Basket Size Distribution")

    # 6. Customer Satisfaction (Ratings)
    plt.subplot(2, 3, 6)
    sns.countplot(x=df["last_review_rating"], palette="magma")
    plt.title("Customer Satisfaction Ratings")

    plt.tight_layout()
    plt.show()

# ------------------------- Main Execution -------------------------
if __name__ == "__main__":
    # Run analysis
    metrics = analyze_data(df)
    df = segment_customers(df)
    model_data = predict_spending(df)

    # Display Key Insights
    print(f"\nLive Insights for {business_type} Business\n")
    print(f"Average Spending: ₹{metrics['avg_spent']:,.2f}")
    print(f"Purchase Frequency: {metrics['avg_frequency']:.1f} times/customer")
    print(f"Churn Risk: {metrics['churn_rate']:.1f}%\n")

    print("\nPredictive Analytics\n")
    print(f"R-squared: {model_data['r2']:.2f} | MAE: ₹{model_data['mae']:,.2f}")
    print(f"Projected Spending for 20 Purchases: ₹{model_data['prediction']:,.2f}\n")

    # AI Insights
    ai_recommendations = generate_ai_insights(metrics, business_type)
    print("\nAI-Powered Recommendations:\n")
    print(ai_recommendations)

    # AI Q&A
    user_question = input("\nAsk AI a business-specific question: ")
    ai_answer = generate_ai_insights(metrics, business_type, user_question)
    print("\nAI Response:\n", ai_answer)

    # Show Dashboard with 6 Graphs
    visualize_dashboard(df, metrics)
