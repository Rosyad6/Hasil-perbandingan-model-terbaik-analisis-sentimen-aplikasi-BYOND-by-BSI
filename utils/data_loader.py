import pandas as pd


def load_data(uploaded_file):

    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)

    elif uploaded_file.name.endswith('.xlsx'):
        return pd.read_excel(uploaded_file)

    elif uploaded_file.name.endswith('.xls'):
        return pd.read_excel(uploaded_file)

    elif uploaded_file.name.endswith('.json'):
        return pd.read_json(uploaded_file)

    elif uploaded_file.name.endswith('.tsv'):
        return pd.read_csv(uploaded_file, sep='\t')
