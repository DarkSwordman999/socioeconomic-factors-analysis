"""
Лабораторная работа С. Описание и построение модели
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scripts.load_data import load_and_prepare_data

def print_model_description():
    """
    Подробное описание модели для отчёта
    """
    print("=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА С")
    print("Описание и построение модели")
    print("=" * 80)
    
    print("\n1. ВЫБОР ТИПА МОДЕЛИ")
    print("-" * 50)
    print("""
    Исходя из цели исследования (изучение взаимосвязей между факторами
    социально-экономического развития) и характера данных, выбрана модель
    множественной линейной регрессии.
    
    Обоснование выбора:
    1. Имеется одна зависимая переменная (социальный показатель) и несколько
       независимых (экономические факторы).
    2. Между переменными предполагается линейная связь (подтверждается
       корреляционным анализом).
    3. Модель позволяет количественно оценить вклад каждого фактора.
    """)
    
    print("\n2. МАТЕМАТИЧЕСКОЕ ОПИСАНИЕ МОДЕЛИ")
    print("-" * 50)
    print("""
    Общий вид множественной линейной регрессии:
    
        Y = β₀ + β₁·X₁ + β₂·X₂ + β₃·X₃ + ε
    
    где:
        Y   - зависимая переменная (социальный показатель)
        X₁  - ВРП на душу населения (руб.)
        X₂  - Инвестиции на душу населения (руб.)
        X₃  - Среднедушевые денежные доходы (руб.)
        β₀  - свободный член (константа)
        β₁, β₂, β₃ - коэффициенты регрессии
        ε   - случайная ошибка (остаток)
    
    Оценка параметров модели производится методом наименьших квадратов (МНК):
    
        Σ(Y_i - Ŷ_i)² → min
    
    Качество модели оценивается с помощью:
        - коэффициента детерминации R²
        - F-критерия Фишера
        - t-критерия Стьюдента для каждого коэффициента
        - средней абсолютной процентной ошибки MAPE
    """)
    
    print("\n3. ПОСТРОЕНИЕ МОДЕЛЕЙ ДЛЯ КАЖДОГО СОЦИАЛЬНОГО ПОКАЗАТЕЛЯ")
    print("-" * 50)
    
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
        
        # Построение модели OLS
        model = sm.OLS(y, X).fit()
        models[target_name] = model
        
        print(f"\n{'='*60}")
        print(f"Модель: {target_name}")
        print(f"{'='*60}")
        print(f"Уравнение регрессии:")
        print(f"  {target_name} = {model.params['const']:.4f}", end="")
        
        if model.params['ВРП'] >= 0:
            print(f" + {model.params['ВРП']:.6f}·ВРП", end="")
        else:
            print(f" - {abs(model.params['ВРП']):.6f}·ВРП", end="")
        
        if model.params['Инвестиции'] >= 0:
            print(f" + {model.params['Инвестиции']:.6f}·Инвестиции", end="")
        else:
            print(f" - {abs(model.params['Инвестиции']):.6f}·Инвестиции", end="")
        
        if model.params['Доходы'] >= 0:
            print(f" + {model.params['Доходы']:.6f}·Доходы", end="")
        else:
            print(f" - {abs(model.params['Доходы']):.6f}·Доходы", end="")
        
        print(f"\n\nКоэффициент детерминации R² = {model.rsquared:.4f}")
        print(f"Скорректированный R² = {model.rsquared_adj:.4f}")
        print(f"F-статистика = {model.fvalue:.2f} (p-value = {model.f_pvalue:.4e})")
        
        # Интерпретация коэффициентов
        print(f"\nИнтерпретация коэффициентов:")
        for var in ['ВРП', 'Инвестиции', 'Доходы']:
            coef = model.params[var]
            pval = model.pvalues[var]
            sign = "положительное" if coef > 0 else "отрицательное"
            significant = "значимо" if pval < 0.05 else "незначимо"
            print(f"  β({var}) = {coef:.6f} ({sign} влияние, p={pval:.4f} -> {significant})")
    
    return models

def analyze_residuals(model, target_name, df, X, y):
    """
    Анализ остатков модели
    """
    residuals = model.resid
    print(f"\nАнализ остатков для модели '{target_name}':")
    print("-" * 40)
    print(f"  Среднее остатков: {residuals.mean():.6f} (должно быть близко к 0)")
    print(f"  Дисперсия остатков: {residuals.var():.4f}")
    print(f"  Асимметрия остатков: {residuals.skew():.4f}")
    print(f"  Эксцесс остатков: {residuals.kurtosis():.4f}")

if __name__ == "__main__":
    models = print_model_description()
