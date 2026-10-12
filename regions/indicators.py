"""Перечень показателей и их расположение в файлах Росстата."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Indicator:
    key: str        # имя столбца в итоговой панели
    section: int    # номер раздела сборника (файл «Раздел N - ….xlsx»)
    sheet: str      # лист с таблицей
    title: str      # название показателя
    unit: str       # единица измерения
    money: bool = False  # рубли в текущих ценах — пересчитываются в реальные


INDICATORS = [
    # экономические факторы
    Indicator("gdp", 8, "8.2.", "ВРП на душу населения", "руб.", money=True),
    Indicator("invest", 10, "10.2.", "Инвестиции в основной капитал на душу населения", "руб.", money=True),
    Indicator("income", 3, "3.2.", "Среднедушевые денежные доходы", "руб. в месяц", money=True),
    # социальные показатели
    Indicator("poverty", 3, "3.12.", "Уровень бедности", "% населения"),
    Indicator("life_exp", 1, "1.16.1.", "Ожидаемая продолжительность жизни", "лет"),
    Indicator("housing", 3, "3.25.", "Жилая площадь на одного жителя", "кв. м"),
    Indicator("infant_mort", 1, "1.13.", "Младенческая смертность", "на 1000 родившихся"),
    # контрольные переменные
    Indicator("unemployment", 2, "2.10.1.", "Уровень безработицы", "%"),
    Indicator("doctors", 5, "5.4.2.", "Врачи на 10 000 человек", "чел."),
    Indicator("urban", 1, "1.3.", "Доля городского населения", "%"),
    Indicator("elderly", 1, "1.6.3.", "Доля населения старше трудоспособного возраста", "%"),
    Indicator("students", 4, "4.19.", "Студенты вузов на 10 000 человек", "чел."),
    # ценовой дефлятор
    Indicator("basket", 20, "20.3.1.", "Стоимость фиксированного набора товаров и услуг", "руб."),
]

BY_KEY = {ind.key: ind for ind in INDICATORS}

MONEY = [ind.key for ind in INDICATORS if ind.money]
FACTORS = ["gdp_real", "invest_real", "income_real"]
TARGETS = ["poverty", "life_exp", "housing"]
CONTROLS = ["unemployment", "doctors", "urban", "elderly"]

LABELS = {ind.key: ind.title for ind in INDICATORS} | {
    "gdp_real": "ВРП на душу (в ценах 2023 г.)",
    "invest_real": "Инвестиции на душу (в ценах 2023 г.)",
    "income_real": "Доходы на душу (в ценах 2023 г.)",
}
UNITS = {ind.key: ind.unit for ind in INDICATORS} | {
    "gdp_real": "руб.",
    "invest_real": "руб.",
    "income_real": "руб. в месяц",
}
