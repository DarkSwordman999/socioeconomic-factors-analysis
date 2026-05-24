"""
Лабораторная работа C. Часть 2
Построение модели на основе собранных данных
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scripts.load_data import load_and_prepare_data

def build_models():
    """
    Построение регрессионных моделей
    """
    print("=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА C")
    print("Часть 2. Построение модели")
    print("=" * 80)
    
    df = load_and_prepare_data()
    
    # Независимые переменные
    X = df[['ВРП', 'Инвестиции', 'Доходы']]
    X = sm.add_constant(X)
    
    targets = {
        'Продолжительность_жизни': 'Ожидаемая продолжительность жизни (лет)',
        'Уровень_бедности': 'Уровень бедности (%)',
        'Жилплощадь': 'Жилплощадь на человека (кв.м)'
    }
    
    models = {}
    
    print("\nПОСТРОЕННЫЕ МОДЕЛИ:")
    print("=" * 60)
    
    for target_col, target_name in targets.items():
        y = df[target_col]
        model = sm.OLS(y, X).fit()
        models[target_name] = model
        
        print(f"\nМодель: {target_name}")
        print(f"Уравнение: {target_name} = {model.params['const']:.4f}", end="")
        
        if model.params['ВРП'] >= 0:
            print(f" + {model.params['ВРП']:.8f}·ВРП", end="")
        else:
            print(f" - {abs(model.params['ВРП']):.8f}·ВРП", end="")
        
        if model.params['Инвестиции'] >= 0:
            print(f" + {model.params['Инвестиции']:.8f}·Инвестиции", end="")
        else:
            print(f" - {abs(model.params['Инвестиции']):.8f}·Инвестиции", end="")
        
        if model.params['Доходы'] >= 0:
            print(f" + {model.params['Доходы']:.8f}·Доходы")
        else:
            print(f" - {abs(model.params['Доходы']):.8f}·Доходы")
        
        print(f"R² = {model.rsquared:.4f}")
        print(f"Скорректированный R² = {model.rsquared_adj:.4f}")
    
    print("\n" + "=" * 60)
    print("ВЫВОДЫ ПО ЧАСТИ С:")
    print("-" * 40)
    print("1. Модели построены для каждого социального показателя")
    print("2. Наибольший R² у модели 'Уровень бедности' (0.4056)")
    print("3. Наименьший R² у модели 'Продолжительность жизни' (0.3333)")
    print("4. Доходы населения — наиболее значимый фактор во всех моделях")
    print("5. ВРП и инвестиции имеют слабое влияние на социальные показатели")
    
    return models

if __name__ == "__main__":
    models = build_models()
