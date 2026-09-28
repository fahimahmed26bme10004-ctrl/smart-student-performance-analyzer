import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class StudentPerformanceAnalyzer:
    """
    A smart analyzer to process, analyze, and visualize student academic performance data.
    """

    def __init__(self, data: pd.DataFrame):
        self.df = data.copy()
        self.subject_cols = []
        self.processed_df = None

    def preprocess_data(self, subject_columns: list):
        """
        Calculates totals, averages, letter grades, and performance categories.
        """
        self.subject_cols = subject_columns

        # Ensure scores are numeric
        for col in self.subject_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)

        # 1. Total and Average Calculation
        self.df['Total Score'] = self.df[self.subject_cols].sum(axis=1)
        self.df['Average Score'] = self.df[self.subject_cols].mean(axis=1)

        # 2. Grade Assignment
        self.df['Grade'] = self.df['Average Score'].apply(self._assign_grade)

        # 3. Status (Pass/Fail based on minimum score in each subject)
        min_pass_score = 40
        self.df['Status'] = self.df[self.subject_cols].apply(
            lambda row: 'Pass' if all(score >= min_pass_score for score in row) else 'Fail',
            axis=1
        )

        # 4. Performance Category
        self.df['Performance Category'] = self.df['Average Score'].apply(self._categorize_performance)

        self.processed_df = self.df
        return self.processed_df

    @staticmethod
    def _assign_grade(score: float) -> str:
        """Assigns letter grades based on average score."""
        if score >= 90:
            return 'A+'
        elif score >= 80:
            return 'A'
        elif score >= 70:
            return 'B'
        elif score >= 60:
            return 'C'
        elif score >= 50:
            return 'D'
        else:
            return 'F'

    @staticmethod
    def _categorize_performance(score: float) -> str:
        """Categorizes students for targeted academic interventions."""
        if score >= 85:
            return 'High Achiever'
        elif score >= 60:
            return 'Moderate / Satisfactory'
        else:
            return 'Needs Support'

    def generate_class_summary(self) -> pd.DataFrame:
        """
        Calculates class-wide summary statistics per subject.
        """
        if self.processed_df is None:
            raise ValueError("Data must be processed first. Call preprocess_data().")

        summary = {}
        for subject in self.subject_cols:
            summary[subject] = {
                'Mean': round(self.processed_df[subject].mean(), 2),
                'Median': round(self.processed_df[subject].median(), 2),
                'Std Dev': round(self.processed_df[subject].std(), 2),
                'Highest': self.processed_df[subject].max(),
                'Lowest': self.processed_df[subject].min()
            }
        
        return pd.DataFrame(summary).T

    def plot_student_vs_average(self, student_id: str, id_column: str = 'Student ID'):
        """
        Generates a bar chart comparing a specific student's marks against the class average.
        """
        student_data = self.processed_df[self.processed_df[id_column] == student_id]

        if student_data.empty:
            print(f"Error: Student ID '{student_id}' not found.")
            return

        student_scores = student_data[self.subject_cols].iloc[0].values
        class_averages = self.processed_df[self.subject_cols].mean().values

        x = np.arange(len(self.subject_cols))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width/2, student_scores, width, label='Student Score', color='#2b5c8f')
        ax.bar(x + width/2, class_averages, width, label='Class Average', color='#d95f02')

        ax.set_ylabel('Scores')
        ax.set_title(f'Performance Analysis: {student_data["Name"].iloc[0]} ({student_id})')
        ax.set_xticks(x)
        ax.set_xticklabels(self.subject_cols)
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

    def export_report(self, filename: str = "performance_report.csv"):
        """Exports processed data to a CSV file."""
        if self.processed_df is not None:
            self.processed_df.to_csv(filename, index=False)
            print(f"Report exported successfully to {filename}")


# ==========================================
# Execution & Sample Usage
# ==========================================
if __name__ == "__main__":
    # Sample Dataset
    raw_data = {
        'Student ID': ['S101', 'S102', 'S103', 'S104', 'S105'],
        'Name': ['Alice Smith', 'Bob Jones', 'Charlie Brown', 'Diana Prince', 'Evan Wright'],
        'Math': [88, 45, 92, 35, 78],
        'Science': [92, 58, 96, 42, 65],
        'English': [85, 62, 89, 50, 70],
        'History': [78, 50, 91, 38, 82]
    }

    subjects = ['Math', 'Science', 'English', 'History']
    df_students = pd.DataFrame(raw_data)

    # Initialize Analyzer
    analyzer = StudentPerformanceAnalyzer(df_students)

    # 1. Process Scores
    processed = analyzer.preprocess_data(subject_columns=subjects)
    print("--- Processed Student Data ---")
    print(processed[['Student ID', 'Name', 'Average Score', 'Grade', 'Status', 'Performance Category']])
    print("\n")

    # 2. Get Class Overview
    print("--- Subject Summary Statistics ---")
    print(analyzer.generate_class_summary())
    print("\n")

    # 3. Export Summary
    analyzer.export_report("student_summary.csv")

    # 4. Generate Visual Comparison for a specific student
    analyzer.plot_student_vs_average(student_id='S101')