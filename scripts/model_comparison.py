"""
Сравнение различных типов регрессионных моделей
Линейная, экспоненциальная, логарифмическая, полиномиальная
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scripts.load_data import load_and_prepare_data

def compare_models():
    """
    Сравнение качества различных моделей
    """
    print("=" * 80)
    print("СРАВНЕНИЕ МОДЕЛЕЙ РАЗЛИЧНЫХ ТИПОВ")
    print("=" * 80)
    
    df = load_and_prepare_data()
    
    # Для примера возьмём зависимость продолжительности жизни от доходов
    X = df[['Доходы']].values
    y = df['Продолжительность_жизни'].values
    
    # 1. Линейная модель
    model_linear = LinearRegression()
    model_linear.fit(X, y)
    y_pred_linear = model_linear.predict(X)
    
    # 2. Экспоненциальная модель (y = a * exp(b*x))
    # Преобразуем: ln(y) = ln(a) + b*x
    y_log = np.log(y)
    model_exp = LinearRegression()
    model_exp.fit(X, y_log)
    y_pred_exp = np.exp(model_exp.predict(X))
    
    # 3. Логарифмическая модель (y = a + b*ln(x))
    X_log = np.log(X)
    model_log = LinearRegression()
    model_log.fit(X_log, y)
    y_pred_log = model_log.predict(X_log)
    
    # 4. Полиномиальная модель 2-й степени
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model_poly = LinearRegression()
    model_poly.fit(X_poly, y)
    y_pred_poly = model_poly.predict(X_poly)
    
    # 5. Полиномиальная модель 3-й степени
    poly3 = PolynomialFeatures(degree=3)
    X_poly3 = poly3.fit_transform(X)
    model_poly3 = LinearRegression()
    model_poly3.fit(X_poly3, y)
    y_pred_poly3 = model_poly3.predict(X_poly3)
    
    # Сбор метрик
    models = {
        'Линейная': y_pred_linear,
        'Экспоненциальная': y_pred_exp,
        'Логарифмическая': y_pred_log,
        'Полиномиальная 2-й степени': y_pred_poly,
        'Полиномиальная 3-й степени': y_pred_poly3
    }
    
    print("\nСравнение моделей (зависимость: Продолжительность жизни от Доходов)")
    print("=" * 80)
    print(f"{'Модель':<30} {'R²':<10} {'MAE':<10} {'RMSE':<10}")
    print("-" * 80)
    
    results = {}
    for name, y_pred in models.items():
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        results[name] = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
        print(f"{name:<30} {r2:.4f}     {mae:.4f}   {rmse:.4f}")
    
    # Визуализация
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    x_sorted = np.sort(X, axis=0)
    
    for idx, (name, y_pred) in enumerate(models.items()):
        axes[idx].scatter(X, y, alpha=0.3, color='steelblue', label='Данные')
        
        # Сортируем для плавной линии
        sorted_idx = np.argsort(X[:, 0])
        X_sorted = X[sorted_idx]
        y_pred_sorted = y_pred[sorted_idx]
        
        axes[idx].plot(X_sorted, y_pred_sorted, 'r-', linewidth=2, label=name)
        axes[idx].set_xlabel('Доходы (руб.)')
        axes[idx].set_ylabel('Продолжительность жизни (лет)')
        axes[idx].set_title(f'{name}\nR² = {results[name]["R2"]:.4f}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    # Удаляем лишний подграфик (6-й)
    axes[5].remove()
    
    plt.tight_layout()
    plt.savefig('results/figures/models_comparison.png', dpi=150)
    plt.close()
    print("\n График сравнения моделей сохранён в results/figures/models_comparison.png")
    
    # Вывод лучшей модели
    print("\n" + "=" * 80)
    print("ВЫВОДЫ ПО СРАВНЕНИЮ МОДЕЛЕЙ")
    print("=" * 80)
    
    best_model = max(results, key=lambda x: results[x]['R2'])
    print(f"\nЛучшая модель по R²: {best_model} (R² = {results[best_model]['R2']:.4f})")
    
    print("\nРекомендация:")
    if best_model == 'Линейная':
        print("  Линейная модель показывает наилучшее качество.")
        print("  Нелинейные модели не дают значительного улучшения.")
    elif best_model == 'Экспоненциальная':
        print("  Экспоненциальная модель показывает лучшее качество.")
        print("  Рекомендуется использовать экспоненциальную регрессию.")
    elif best_model == 'Логарифмическая':
        print("  Логарифмическая модель показывает лучшее качество.")
        print("  Рекомендуется использовать логарифмическую регрессию.")
    else:
        print(f"  {best_model} показывает наилучшее качество.")
        print("  Рекомендуется использовать полиномиальную регрессию.")

if __name__ == "__main__":
    compare_models()
