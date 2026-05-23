import pandas as pd

def load_gdp():
    df = pd.read_excel('data/raw/Раздел 8 - Валовой региональный продукт.xlsx', sheet_name='8.2.', skiprows=3, header=None)
    years = df.iloc[2, 1:].dropna().values
    df = df.iloc[3:, :].copy()
    df.columns = ['Регион'] + list(years)
    df = df.dropna(subset=['Регион'])
    df_long = df.melt(id_vars=['Регион'], var_name='Год', value_name='ВРП')
    df_long = df_long.dropna(subset=['ВРП'])
    df_long['Год'] = df_long['Год'].astype(int)
    return df_long

if __name__ == "__main__":
    data = load_gdp()
    print(f"GDP shape: {data.shape}")
