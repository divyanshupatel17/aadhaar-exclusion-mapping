"""
Visualization Utilities
Create publication-quality charts for analysis and reporting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional


# Set default matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_enrollment_by_state(df: pd.DataFrame, 
                               top_n: int = 15, 
                               figsize: tuple = (12, 8),
                               save_path: Optional[str] = None):
    """
    Plot top states by total enrollments (horizontal bar chart).
    
    Args:
        df: Master district DataFrame
        top_n: Number of top states to show
        figsize: Figure size (width, height)
        save_path: Path to save figure (if None, doesn't save)
    """
    state_summary = df.groupby('state')['total_enrollments'].sum().sort_values(ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    state_summary.plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')
    ax.set_title(f'Top {top_n} States by Aadhaar Enrollments', fontsize=16, weight='bold')
    ax.set_xlabel('Total Enrollments', fontsize=12)
    ax.set_ylabel('State', fontsize=12)
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved: {save_path}")
    
    plt.show()


def plot_age_distribution_pie(df: pd.DataFrame,
                                figsize: tuple = (10, 8),
                                save_path: Optional[str] = None):
    """
    Plot national age distribution as pie chart.
    
    Args:
        df: Enrolment DataFrame with age columns
        figsize: Figure size
        save_path: Path to save figure
    """
    age_data = {
        'Children (0-5)': df['age_0_5'].sum(),
        'Children (5-17)': df['age_5_17'].sum(),
        'Adults (18+)': df['age_18_greater'].sum()
    }
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.pie(age_data.values(), labels=age_data.keys(), autopct='%1.1f%%',
           colors=['#FF6B6B', '#4ECDC4', '#45B7D1'], startangle=90)
    ax.set_title('National Age Distribution - Aadhaar Enrollments', fontsize=16, weight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved: {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, 
                           labels: list = ['Low Risk', 'High Risk'],
                           figsize: tuple = (8, 6),
                           save_path: Optional[str] = None):
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        figsize: Figure size
        save_path: Path to save figure
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'}, ax=ax)
    ax.set_title('Confusion Matrix - Exclusion Risk Prediction', fontsize=16, weight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved: {save_path}")
    
    plt.show()


def plot_feature_importance(importance_df: pd.DataFrame,
                              figsize: tuple = (10, 8),
                              save_path: Optional[str] = None):
    """
    Plot feature importance horizontal bar chart.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        figsize: Figure size
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
    ax.barh(importance_df['feature'], importance_df['importance'], 
            color=colors, edgecolor='black')
    
    ax.set_xlabel('Importance Score', fontsize=14, weight='bold')
    ax.set_ylabel('Feature', fontsize=14, weight='bold')
    ax.set_title('Feature Importance - Gradient Boosting Model', fontsize=16, weight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved: {save_path}")
    
    plt.show()


def plot_roi_analysis(df_priority: pd.DataFrame,
                       figsize: tuple = (14, 6),
                       save_path: Optional[str] = None):
    """
    Plot ROI distribution and cost vs benefit scatter.
    
    Args:
        df_priority: Priority districts DataFrame with cost/benefit columns
        figsize: Figure size
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # ROI histogram
    axes[0].hist(df_priority['roi_percentage'], bins=30, color='green', 
                 edgecolor='black', alpha=0.7)
    axes[0].axvline(df_priority['roi_percentage'].mean(), color='red', 
                    linestyle='--', linewidth=2, 
                    label=f'Mean: {df_priority["roi_percentage"].mean():.1f}%')
    axes[0].set_title('ROI Distribution', fontsize=14, weight='bold')
    axes[0].set_xlabel('ROI (%)', fontsize=12)
    axes[0].set_ylabel('Number of Districts', fontsize=12)
    axes[0].legend()
    
    # Cost vs Benefit scatter
    scatter = axes[1].scatter(df_priority['total_cost']/100000, 
                               df_priority['net_benefit']/100000,
                               c=df_priority['priority_score'], 
                               cmap='Reds', s=100, alpha=0.6)
    axes[1].set_title('Cost vs Net Benefit', fontsize=14, weight='bold')
    axes[1].set_xlabel('Total Cost (₹ Lakhs)', fontsize=12)
    axes[1].set_ylabel('Net Benefit (₹ Lakhs)', fontsize=12)
    axes[1].grid(alpha=0.3)
    plt.colorbar(scatter, ax=axes[1], label='Priority Score')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved: {save_path}")
    
    plt.show()


def create_interactive_risk_map(df: pd.DataFrame, 
                                  output_path: str):
    """
    Create interactive Plotly scatter plot of exclusion risk.
    
    Args:
        df: Master district DataFrame
        output_path: Path to save HTML file
    """
    fig = px.scatter(df, 
                     x='total_enrollments', 
                     y='child_enrollment_rate',
                     color='exclusion_risk_score',
                     hover_data=['state', 'district'],
                     title='Exclusion Risk Map: Enrollment vs Child Rate',
                     labels={'total_enrollments': 'Total Enrollments',
                             'child_enrollment_rate': 'Child Enrollment Rate',
                             'exclusion_risk_score': 'Risk Score'},
                     color_continuous_scale='Reds')
    
    fig.update_layout(height=700)
    fig.write_html(output_path)
    print(f"Interactive chart saved: {output_path}")


def create_dashboard(df_master: pd.DataFrame,
                      df_priority: pd.DataFrame,
                      output_path: str):
    """
    Create comprehensive Plotly dashboard with multiple panels.
    
    Args:
        df_master: Master district DataFrame
        df_priority: Priority districts DataFrame
        output_path: Path to save HTML dashboard
    """
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Exclusion Risk Distribution', 
                        'ROI by State (Top 10)',
                        'Phase-wise Budget Allocation',
                        'People Reached by Phase'),
        specs=[[{'type': 'histogram'}, {'type': 'bar'}],
               [{'type': 'pie'}, {'type': 'bar'}]]
    )
    
    # Panel 1: Risk distribution
    fig.add_trace(
        go.Histogram(x=df_master['exclusion_risk_score'], nbinsx=50, 
                     name='Risk Score', marker_color='coral'),
        row=1, col=1
    )
    
    # Panel 2: ROI by state
    state_roi = df_priority.groupby('state')['roi_percentage'].mean().sort_values(ascending=False).head(10)
    fig.add_trace(
        go.Bar(x=state_roi.values, y=state_roi.index, orientation='h', marker_color='teal'),
        row=1, col=2
    )
    
    # Panel 3: Budget pie
    phase_budget = df_priority.groupby('deployment_phase')['total_cost'].sum()
    fig.add_trace(
        go.Pie(labels=phase_budget.index, values=phase_budget.values, 
               marker=dict(colors=['#d62728', '#ff7f0e', '#2ca02c'])),
        row=2, col=1
    )
    
    # Panel 4: People reached
    phase_people = df_priority.groupby('deployment_phase')['estimated_new_enrollments'].sum()
    fig.add_trace(
        go.Bar(x=phase_people.index, y=phase_people.values, 
               marker_color=['#d62728', '#ff7f0e', '#2ca02c']),
        row=2, col=2
    )
    
    fig.update_layout(height=900, showlegend=False, 
                       title_text="Aadhaar Exclusion - Comprehensive Dashboard", 
                       title_font_size=24)
    fig.write_html(output_path)
    print(f"Comprehensive dashboard saved: {output_path}")
