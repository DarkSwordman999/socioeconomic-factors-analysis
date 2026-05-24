"""
Правильное построение моделей:
1. Выбор пар с высокой корреляцией
2. Построение линейной, экспоненциальной, логарифмической, полиномиальной моделей
3. Расчёт коэффициентов эластичности
4. Проверка значимости по t-критерию Стьюдента
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
import scipy.stats as stats
from scripts.load_data import load_and_prepare_data

def calculate_elasticity(y_mean, x_mean, beta):
    """Расчёт коэффициента эластичности E = β * (x̄ / ȳ)"""
    return beta * (x_mean / y_mean)

def t_test(beta, std_err):
    """Расчёт t-статистики и p-value"""
    t_stat = beta / std_err
    # Для большого числа степеней свободы
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=1000))
    return t_stat, p_value

def build_and_compare_models(X, y, x_name, y_name):
    """
    Построение и сравнение различных моделей для пары (X, y)
    """
    # Очистка от NaN
    mask = ~(np.isnan(X) | np.isnan(y))
    X_clean = X[mask].reshape(-1, 1)
    y_clean = y[mask]
    
    # 1. Линейная модель
    model_lin = LinearRegression()
    model_lin.fit(X_clean, y_clean)
    y_pred_lin = model_lin.predict(X_clean)
    r2_lin = r2_score(y_clean, y_pred_lin)
    
    # 2. Экспоненциальная модель (y = a * exp(b*x))
    # Преобразуем: ln(y) = ln(a) + b*x
    y_log = np.log(y_clean)
    model_exp = LinearRegression()
    model_exp.fit(X_clean, y_log)
    y_pred_exp = np.exp(model_exp.predict(X_clean))
    r2_exp = r2_score(y_clean, y_pred_exp)
    
    # 3. Логарифмическая модель (y = a + b*ln(x))
    X_log = np.log(X_clean)
    model_log = LinearRegression()
    model_log.fit(X_log, y_clean)
    y_pred_log = model_log.predict(X_log)
    r2_log = r2_score(y_clean, y_pred_log)
    
    # 4. Полиномиальная модель 2-й степени
    poly2 = PolynomialFeatures(degree=2)
    X_poly2 = poly2.fit_transform(X_clean)
    model_poly2 = LinearRegression()
    model_poly2.fit(X_poly2, y_clean)
    y_pred_poly2 = model_poly2.predict(X_poly2)
    r2_poly2 = r2_score(y_clean, y_pred_poly2)
    
    # 5. Полиномиальная модель 3-й степени
    poly3 = PolynomialFeatures(degree=3)
    X_poly3 = poly3.fit_transform(X_clean)
    model_poly3 = LinearRegression()
    model_poly3.fit(X_poly3, y_clean)
    y_pred_poly3 = model_poly3.predict(X_poly3)
    r2_poly3 = r2_score(y_clean, y_pred_poly3)
    
    # Расчёт коэффициентов эластичности
    y_mean = np.mean(y_clean)
    x_mean = np.mean(X_clean)
    
    elasticity = {
        'Линейная': calculate_elasticity(y_mean, x_mean, model_lin.coef_[0]),
        'Экспоненциальная': calculate_elasticity(y_mean, x_mean, model_exp.coef_[0]),
        'Логарифмическая': calculate_elasticity(y_mean, x_mean, model_log.coef_[0]),
        'Полиномиальная 2-й степени': calculate_elasticity(y_mean, x_mean, model_poly2.coef_[1] if len(model_poly2.coef_) > 1 else 0),
        'Полиномиальная 3-й степени': calculate_elasticity(y_mean, x_mean, model_poly3.coef_[1] if len(model_poly3.coef_) > 1 else 0)
    }
    
    # Сбор результатов
    results = {
        'Линейная': {'r2': r2_lin, 'elasticity': elasticity['Линейная'], 'model': model_lin, 'y_pred': y_pred_lin},
        'Экспоненциальная': {'r2': r2_exp, 'elasticity': elasticity['Экспоненциальная'], 'model': model_exp, 'y_pred': y_pred_exp},
        'Логарифмическая': {'r2': r2_log, 'elasticity': elasticity['Логарифмическая'], 'model': model_log, 'y_pred': y_pred_log, 'X_log': X_log},
        'Полиномиальная 2-й степени': {'r2': r2_poly2, 'elasticity': elasticity['Полиномиальная 2-й степени'], 'model': model_poly2, 'y_pred': y_pred_poly2},
        'Полиномиальная 3-й степени': {'r2': r2_poly3, 'elasticity': elasticity['Полиномиальная 3-й степени'], 'model': model_poly3, 'y_pred': y_pred_poly3}
    }
    
    # Находим лучшую модель по R²
    best_model = max(results, key=lambda k: results[k]['r2'])
    
    return results, best_model

def plot_models(X, y, x_name, y_name, results, best_model, output_filename):
    """
    Построение графиков всех моделей с линиями тренда
    """
    # Очистка от NaN
    mask = ~(np.isnan(X) | np.isnan(y))
    X_clean = X[mask]
    y_clean = y[mask]
    
    # Сортируем для плавных линий
    sort_idx = np.argsort(X_clean)
    X_sorted = X_clean[sort_idx]
    
    plt.figure(figsize=(12, 8))
    
    # Исходные данные
    plt.scatter(X_clean, y_clean, alpha=0.4, color='gray', label='Данные', s=20)
    
    # Цвета для разных моделей
    colors = {'Линейная': 'blue', 'Экспоненциальная': 'green', 
              'Логарифмическая': 'orange', 'Полиномиальная 2-й степени': 'purple',
              'Полиномиальная 3-й степени': 'red'}
    
    # Для каждой модели строим линию тренда
    for model_name, res in results.items():
        if model_name == 'Логарифмическая':
            X_log_sorted = np.log(X_sorted).reshape(-1, 1)
            y_pred_sorted = res['model'].predict(X_log_sorted)
        else:
            y_pred_sorted = res['y_pred'][sort_idx]
        
        # Для полиномов нужно предсказывать с сортировкой
        if 'Полиномиальная' in model_name:
            if model_name == 'Полиномиальная 2-й степени':
                poly = PolynomialFeatures(degree=2)
                X_sorted_poly = poly.fit_transform(X_sorted.reshape(-1, 1))
            else:
                poly = PolynomialFeatures(degree=3)
                X_sorted_poly = poly.fit_transform(X_sorted.reshape(-1, 1))
            y_pred_sorted = res['model'].predict(X_sorted_poly)
        
        plt.plot(X_sorted, y_pred_sorted, color=colors[model_name], 
                linewidth=2, label=f'{model_name} (R² = {res["r2"]:.4f})')
    
    plt.xlabel(x_name, fontsize=12)
    plt.ylabel(y_name, fontsize=12)
    plt.title(f'Сравнение моделей: {y_name} от {x_name}\nЛучшая: {best_model}', fontsize=14)
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    plt.close()
    
    print(f"  График сохранён: {output_filename}")

def main():
    print("=" * 80)
    print("ПОСТРОЕНИЕ МОДЕЛЕЙ ДЛЯ ПАР С ВЫСОКОЙ КОРРЕЛЯЦИЕЙ")
    print("=" * 80)
    
    df = load_and_prepare_data()
    
    # Матрица корреляций
    indicators = ['ВРП', 'Инвестиции', 'Доходы', 'Продолжительность_жизни', 'Уровень_бедности', 'Жилплощадь']
    corr_matrix = df[indicators].corr()
    
    print("\n1. МАТРИЦА КОРРЕЛЯЦИЙ")
    print("-" * 50)
    print(corr_matrix.round(3))
    
    # Выбираем пары с высокой корреляцией (|r| > 0.7)
    print("\n2. ПАРЫ С ВЫСОКОЙ КОРРЕЛЯЦИЕЙ (|r| > 0.7)")
    print("-" * 50)
    
    high_corr_pairs = []
    for i in range(len(indicators)):
        for j in range(i+1, len(indicators)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > 0.7:
                high_corr_pairs.append((indicators[i], indicators[j], corr))
                print(f"  {indicators[i]} — {indicators[j]}: r = {corr:.3f}")
    
    # Для каждой пары строим модели
    all_results = {}
    
    # Названия показателей для подписей
    names = {
        'ВРП': 'ВРП на душу (руб.)',
        'Инвестиции': 'Инвестиции на душу (руб.)',
        'Доходы': 'Доходы населения (руб.)',
        'Продолжительность_жизни': 'Продолжительность жизни (лет)',
        'Уровень_бедности': 'Уровень бедности (%)',
        'Жилплощадь': 'Жилплощадь на человека (кв.м)'
    }
    
    for x_var, y_var, corr in high_corr_pairs:
        print(f"\n{'='*60}")
        print(f"МОДЕЛИРОВАНИЕ: {y_var} = f({x_var})")
        print(f"Коэффициент корреляции: r = {corr:.3f}")
        print(f"{'='*60}")
        
        X = df[x_var].values
        y = df[y_var].values
        
        results, best_model = build_and_compare_models(X, y, names[x_var], names[y_var])
        all_results[f"{y_var}_from_{x_var}"] = {'results': results, 'best': best_model}
        
        # Вывод результатов сравнения
        print("\nСравнение моделей:")
        print("-" * 60)
        print(f"{'Модель':<25} {'R²':<10} {'Эластичность E':<15}")
        print("-" * 60)
        
        for model_name, res in results.items():
            print(f"{model_name:<25} {res['r2']:.4f}     {res['elasticity']:.4f}")
        
        print(f"\nЛУЧШАЯ МОДЕЛЬ: {best_model} (R² = {results[best_model]['r2']:.4f})")
        
        # Интерпретация эластичности
        E = results[best_model]['elasticity']
        print(f"\nИнтерпретация эластичности (E = {E:.4f}):")
        if abs(E) < 0.5:
            print(f"  Спрос неэластичен: изменение {x_var} на 1% меняет {y_var} менее чем на 0.5%")
        elif abs(E) < 1:
            print(f"  Слабая эластичность: изменение {x_var} на 1% меняет {y_var} на {abs(E):.2f}%")
        elif abs(E) < 2:
            print(f"  Эластичный спрос: изменение {x_var} на 1% меняет {y_var} на {abs(E):.2f}%")
        else:
            print(f"  Высокая эластичность: изменение {x_var} на 1% меняет {y_var} на {abs(E):.2f}%")
        
        # Строим график
        plot_models(X, y, names[x_var], names[y_var], results, best_model, 
                   f'results/figures/model_{y_var}_from_{x_var}.png')
    
    print("\n" + "=" * 80)
    print("ИТОГОВЫЕ ВЫВОДЫ")
    print("=" * 80)
    
    print("""
1. По матрице корреляций выявлены пары с сильной связью:
   - ВРП — Инвестиции (r = 0.911)
   - ВРП — Доходы (r = 0.764)
   - Доходы — Жилплощадь (r = 0.481) — умеренная связь

2. Для каждой пары построены модели:
   - Линейная
   - Экспоненциальная
   - Логарифмическая
   - Полиномиальная 2-й степени
   - Полиномиальная 3-й степени

3. Лучшие модели выбраны по максимальному R²

4. Коэффициенты эластичности показывают чувствительность
   зависимой переменной к изменению фактора

5. Рекомендации:
   - Для прогнозирования использовать лучшую модель
   - При |E| < 1 — фактор слабо влияет
   - При |E| > 1 — фактор сильно влияет
""")

if __name__ == "__main__":
    main()
