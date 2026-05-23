import pandas as pd

def load_investments():
    df = pd.read_excel('data/raw/Раздел 10 - Инвестиции.xlsx', sheet_name='10.2.', skiprows=6, header=None)
    years = df.iloc[0, :].dropna().values
    df = df.iloc[1:, :].copy()
    df.columns = ['Регион'] + list(years)
    df = df.dropna(subset=['Регион'])
    df_long = df.melt(id_vars=['Регион'], var_name='Год', value_name='Инвестиции')
    df_long = df_long.dropna(subset=['Инвестиции'])
    df_long['Год'] = df_long['Год'].astype(int)
    return df_long

if __name__ == "__main__":
    data = load_investments()
    print(f"Investments shape: {data.shape}")
