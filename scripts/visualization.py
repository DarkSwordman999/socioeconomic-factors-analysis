import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats
from scripts.load_data import load_and_prepare_data

def create_visualizations(df):
    """Создание всех графиков для отчёта"""
    
    indicators = ['ВРП', 'Инвестиции', 'Доходы', 'Уровень_бедности', 'Продолжительность_жизни', 'Жилплощадь']
    names = ['ВРП на душу (руб.)', 'Инвестиции на душу (руб.)', 'Среднедушевые доходы (руб.)',
             'Уровень бедности (%)', 'Ожидаемая продолжительность жизни (лет)', 'Жилплощадь на человека (кв.м)']
    
    # 1. Гистограммы с кривой плотности
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (col, name) in enumerate(zip(indicators, names)):
        sns.histplot(df[col], bins=30, kde=True, ax=axes[i], color='steelblue', edgecolor='black')
        axes[i].set_title(f'Распределение: {name}', fontsize=10)
        axes[i].set_xlabel(name)
        axes[i].set_ylabel('Частота')
    
    plt.tight_layout()
    plt.savefig('results/figures/distributions_histograms.png', dpi=150)
    plt.close()
    print("Гистограммы сохранены")
    
    # 2. Ящики с усами (boxplots)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (col, name) in enumerate(zip(indicators, names)):
        axes[i].boxplot(df[col], vert=True, patch_artist=True, 
                        boxprops=dict(facecolor='steelblue', alpha=0.7))
        axes[i].set_title(f'Ящик с усами: {name}', fontsize=10)
        axes[i].set_ylabel(name)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/boxplots.png', dpi=150)
    plt.close()
    print("Boxplots сохранены")
    
    # 3. Q-Q plots для проверки нормальности
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (col, name) in enumerate(zip(indicators, names)):
        stats.probplot(df[col].dropna(), dist="norm", plot=axes[i])
        axes[i].set_title(f'Q-Q plot: {name}', fontsize=10)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/qq_plots.png', dpi=150)
    plt.close()
    print("Q-Q plots сохранены")
    
    # 4. Матрица корреляций
    plt.figure(figsize=(10, 8))
    corr_matrix = df[indicators].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True)
    plt.title('Матрица корреляций социально-экономических показателей')
    plt.tight_layout()
    plt.savefig('results/figures/correlation_matrix.png', dpi=150)
    plt.close()
    print("Матрица корреляций сохранена")

if __name__ == "__main__":
    df = load_and_prepare_data()
    create_visualizations(df)
    print("\nВсе графики сохранены в папку results/figures/")
