import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import cohere

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget,
    QLineEdit, QTextEdit, QMessageBox, QHBoxLayout, QGridLayout, QSizePolicy
)
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Initialize Cohere Client
co = cohere.Client('KTn7ndyWTyFbx9yGwzrS27JYOy0TRjttcObYzk5t')

class DashboardCanvas(FigureCanvas):
    """Matplotlib canvas integrated into PyQt"""
    def __init__(self, parent=None, width=10, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(DashboardCanvas, self).__init__(fig)

class BusinessAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📊 AI-Powered Business Analytics Dashboard")
        self.setGeometry(100, 100, 1200, 800)

        # Set a modern color scheme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QLabel {
                font-size: 14px;
                color: #333333;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QLineEdit, QTextEdit {
                border: 1px solid #cccccc;
                border-radius: 5px;
                padding: 5px;
                font-size: 14px;
            }
            QTextEdit {
                background-color: #ffffff;
                color: #000000;
            }
        """)

        # DataFrame Initialization
        self.df = self.create_sample_data()

        # Main Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main Layout
        self.main_layout = QVBoxLayout(self.central_widget)

        # Header
        header = QLabel("AI-Powered Business Analytics Dashboard")
        header.setFont(QFont("Arial", 18, QFont.Bold))
        header.setStyleSheet("color: #4CAF50;")
        self.main_layout.addWidget(header)

        # Input Section
        self.business_label = QLabel("Enter Business Type:")
        self.business_input = QLineEdit()
        self.analyze_button = QPushButton("Analyze & Generate Insights")
        self.analyze_button.clicked.connect(self.run_analysis)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.business_label)
        input_layout.addWidget(self.business_input)
        input_layout.addWidget(self.analyze_button)
        self.main_layout.addLayout(input_layout)

        # Insights Section
        self.insights_box = QTextEdit()
        self.insights_box.setReadOnly(True)
        self.main_layout.addWidget(QLabel("📄 Key Insights:"))
        self.main_layout.addWidget(self.insights_box)

        # AI Question Section
        self.ai_question_input = QLineEdit()
        self.ai_question_input.setPlaceholderText("Ask AI a Business-Specific Question")
        self.ai_answer_button = QPushButton("Ask AI")
        self.ai_answer_button.clicked.connect(self.ask_ai)

        ai_layout = QHBoxLayout()
        ai_layout.addWidget(self.ai_question_input)
        ai_layout.addWidget(self.ai_answer_button)
        self.main_layout.addLayout(ai_layout)

        # Dashboard Visualization
        self.dashboard_canvas = DashboardCanvas(self, width=10, height=6)
        self.main_layout.addWidget(QLabel("📈 Dashboard Visualization:"))
        self.main_layout.addWidget(self.dashboard_canvas)

        # Footer
        footer = QLabel("© 2025 Business Analytics Inc. | All Rights Reserved")
        footer.setFont(QFont("Arial", 10))
        footer.setStyleSheet("color: #777777;")
        self.main_layout.addWidget(footer)

        # Set size policies for responsiveness
        self.insights_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.dashboard_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def create_sample_data(self):
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
        return pd.DataFrame(data)

    def run_analysis(self):
        business_type = self.business_input.text().strip()
        if not business_type:
            QMessageBox.warning(self, "Input Error", "Please enter a business type.")
            return

        # Data Analysis
        metrics = self.analyze_data(self.df)
        self.df = self.segment_customers(self.df)
        model_data = self.predict_spending(self.df)

        # Display Key Insights
        insights = f"Business Type: {business_type}\n"
        insights += f"Average Spending: ₹{metrics['avg_spent']:,.2f}\n"
        insights += f"Purchase Frequency: {metrics['avg_frequency']:.1f} times/customer\n"
        insights += f"Churn Risk: {metrics['churn_rate']:.1f}%\n"
        insights += f"R-squared: {model_data['r2']:.2f} | MAE: ₹{model_data['mae']:,.2f}\n"
        insights += f"Projected Spending for 20 Purchases: ₹{model_data['prediction']:,.2f}\n"

        # AI Recommendations
        ai_text = self.generate_ai_insights(metrics, business_type)
        insights += "\n📌 AI Recommendations:\n" + ai_text

        self.insights_box.setPlainText(insights)
        self.plot_dashboard(self.df, business_type)

    def analyze_data(self, df):
        return {
            "avg_spent": df["total_spent"].mean(),
            "avg_frequency": df["purchase_frequency"].mean(),
            "churn_rate": (df["last_purchase_days"] > 90).mean() * 100,
            "high_basket_customers": (df["avg_basket_size"] > 5).sum(),
            "low_rated_customers": (df["last_review_rating"] < 3).sum()
        }

    def segment_customers(self, df):
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

    def predict_spending(self, df):
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

    def generate_ai_insights(self, metrics, business_type, question=None):
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

    def ask_ai(self):
        question = self.ai_question_input.text().strip()
        if not question:
            QMessageBox.warning(self, "Input Error", "Please type a question.")
            return
        business_type = self.business_input.text().strip()
        metrics = self.analyze_data(self.df)
        ai_response = self.generate_ai_insights(metrics, business_type, question)
        QMessageBox.information(self, "AI Response", ai_response)

    def plot_dashboard(self, df, business_type):
        self.dashboard_canvas.figure.clear()
        axes = self.dashboard_canvas.figure.subplots(2, 3)

        # Pie Chart - Customer Segmentation
        df.segment_label.value_counts().plot.pie(autopct='%1.1f%%', ax=axes[0, 0], colors=['skyblue', 'orange', 'lightgreen'])
        axes[0, 0].set_title("Customer Segmentation", fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel("")

        # Scatter Plot - Spending vs Purchase Frequency
        sns.scatterplot(x=df["purchase_frequency"], y=df["total_spent"], hue=df["segment_label"], palette="Set2", s=80, ax=axes[0, 1])
        axes[0, 1].set_title("Spending vs Purchase Frequency", fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel("Purchase Frequency")
        axes[0, 1].set_ylabel("Total Spent")

        # Bar Plot - Avg Basket Size by Segment
        df.groupby("segment_label")["avg_basket_size"].mean().plot.bar(color='teal', ax=axes[0, 2])
        axes[0, 2].set_title("Avg Basket Size by Segment", fontsize=12, fontweight='bold')
        axes[0, 2].set_ylabel("Avg Basket Size")
        axes[0, 2].tick_params(axis='x', rotation=20)

        # Line Plot - Age vs Spending Trend
        df_sorted = df.sort_values("customer_age")
        axes[1, 0].plot(df_sorted["customer_age"], df_sorted["total_spent"], marker='o', color='purple')
        axes[1, 0].set_title("Age vs Spending Trend", fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel("Customer Age")
        axes[1, 0].set_ylabel("Total Spent")

        # Histogram - Review Ratings
        sns.histplot(df["last_review_rating"], bins=5, kde=False, color='coral', ax=axes[1, 1])
        axes[1, 1].set_title("Distribution of Review Ratings", fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel("Review Rating")
        axes[1, 1].set_ylabel("Count")

        # Empty plot for spacing
        axes[1, 2].axis('off')

        self.dashboard_canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BusinessAnalyzer()
    window.show()
    sys.exit(app.exec_())