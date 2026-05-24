"""
Лабораторная работа С. Визуализация построенных моделей
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scripts.load_data import load_and_prepare_data

def visualize_models():
    """
    Визуализация регрессионных моделей
    """
    print("=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА С (ВИЗУАЛИЗАЦИЯ МОДЕЛЕЙ)")
    print("=" * 80)
    
    # Загружаем данные
    df = load_and_prepare_data()
    
    # Независимые переменные
    X = df[['ВРП', 'Инвестиции', 'Доходы']]
    X = sm.add_constant(X)
    
    # Список зависимых переменных
    targets = {
        'Продолжительность_жизни': 'Ожидаемая продолжительность жизни (лет)',
        'Уровень_бедности': 'Уровень бедности (%)',
        'Жилплощадь': 'Жилплощадь на человека (кв.м)'
    }
    
    models = {}
    
    for target_col, target_name in targets.items():
        y = df[target_col]
        model = sm.OLS(y, X).fit()
        models[target_name] = model
        print(f" Модель построена: {target_name}")
    
    # 1. ГРАФИК: Фактические vs Предсказанные значения
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (target_name, model) in enumerate(models.items()):
        y_true = model.model.endog
        y_pred = model.fittedvalues
        
        axes[idx].scatter(y_true, y_pred, alpha=0.5, color='steelblue', edgecolors='white', linewidth=0.5)
        
        # Линия идеального прогноза (y = x)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Идеальный прогноз')
        
        axes[idx].set_xlabel('Фактические значения', fontsize=10)
        axes[idx].set_ylabel('Предсказанные значения', fontsize=10)
        axes[idx].set_title(f'{target_name}\nR² = {model.rsquared:.4f}', fontsize=12)
        axes[idx].legend(loc='lower right')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/models_actual_vs_predicted.png', dpi=150)
    plt.close()
    print("\n График 1: Фактические vs Предсказанные значения сохранён")
    
    # 2. ГРАФИК: Остатки модели
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (target_name, model) in enumerate(models.items()):
        residuals = model.resid
        y_pred = model.fittedvalues
        
        axes[idx].scatter(y_pred, residuals, alpha=0.5, color='steelblue', edgecolors='white', linewidth=0.5)
        axes[idx].axhline(y=0, color='r', linestyle='--', linewidth=2)
        
        axes[idx].set_xlabel('Предсказанные значения', fontsize=10)
        axes[idx].set_ylabel('Остатки', fontsize=10)
        axes[idx].set_title(f'Остатки модели: {target_name}', fontsize=12)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/models_residuals.png', dpi=150)
    plt.close()
    print("График 2: Остатки моделей сохранён")
    
    # 3. ГРАФИК: Q-Q plot остатков (проверка нормальности)
    import scipy.stats as stats
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (target_name, model) in enumerate(models.items()):
        residuals = model.resid
        stats.probplot(residuals, dist="norm", plot=axes[idx])
        axes[idx].set_title(f'Q-Q plot остатков: {target_name}', fontsize=12)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/models_qq_plots.png', dpi=150)
    plt.close()
    print("График 3: Q-Q plots остатков сохранён")
    
    # 4. ГРАФИК: Гистограмма остатков с кривой плотности
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (target_name, model) in enumerate(models.items()):
        residuals = model.resid
        sns.histplot(residuals, bins=30, kde=True, ax=axes[idx], color='steelblue', edgecolor='black')
        axes[idx].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[idx].set_xlabel('Остатки', fontsize=10)
        axes[idx].set_ylabel('Частота', fontsize=10)
        axes[idx].set_title(f'Распределение остатков: {target_name}', fontsize=12)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/models_residuals_histograms.png', dpi=150)
    plt.close()
    print("График 4: Гистограммы остатков сохранены")
    
    # 5. ГРАФИК: Влияние каждого фактора на зависимую переменную
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    factors = ['ВРП', 'Инвестиции', 'Доходы']
    factor_names = ['ВРП на душу (руб.)', 'Инвестиции на душу (руб.)', 'Доходы населения (руб.)']
    
    for i, (target_name, model) in enumerate(models.items()):
        for j, (factor, factor_name) in enumerate(zip(factors, factor_names)):
            axes[i, j].scatter(df[factor], model.model.endog, alpha=0.3, color='steelblue', label='Данные')
            
            # Линия тренда
            z = np.polyfit(df[factor], model.model.endog, 1)
            p = np.poly1d(z)
            x_sorted = np.sort(df[factor])
            axes[i, j].plot(x_sorted, p(x_sorted), 'r-', linewidth=2, label='Линейный тренд')
            
            axes[i, j].set_xlabel(factor_name, fontsize=9)
            axes[i, j].set_ylabel(target_name, fontsize=9)
            axes[i, j].set_title(f'{target_name} vs {factor}', fontsize=10)
            axes[i, j].legend(fontsize=8)
            axes[i, j].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/models_factors_influence.png', dpi=150)
    plt.close()
    print("✅ График 5: Влияние факторов на зависимые переменные сохранён")
    
    print("\n" + "=" * 80)
    print("Все графики сохранены в папку results/figures/")
    print("=" * 80)
    
    return models

if __name__ == "__main__":
    models = visualize_models()
