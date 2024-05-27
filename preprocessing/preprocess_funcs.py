import pandas as pd
import numpy as np
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, RobustScaler, MaxAbsScaler, PowerTransformer


def hybrid_balancing(X: pd.DataFrame, y: pd.Series, auto: bool=True, tomek: str='majority', smote: str='not majority') -> pd.DataFrame:
    """
    Retorna um DataFrame com as classes balanceadas das seguinte forma:
    1. Reduz a classe majoritária usando ligacoes de TOMEK (remover instancias que nao adicionam muita informacao).
    2. Equaliza as demais classes usando SMOTE para ficarem com a mesma quantidade de instancias que a classe majoritaria.

    :param X: DataFrame com as variaveis independentes
    :X type: pd.DataFrame
    :param y: Series com a variavel dependente
    :y type: pd.Series
    :param tomek: tipo de undersampling a ser feito pelo TOMEK
    :tomek type: str
    :param smote: tipo de oversampling a ser feito pelo SMOTE
    :smote type: str
    :return: DataFrame transformado
    :rtype: pd.Dataframe
    """


    tl = TomekLinks(sampling_strategy=tomek)
    X_tl, y_tl = tl.fit_resample(X, y)

    if auto:
        smote = y_tl.value_counts().to_dict()
        smote['Dropout'] = smote['Graduate_or_Enrolled']

    sm = SMOTE(sampling_strategy=smote)
    X_tl_sm, y_tl_sm = sm.fit_resample(X_tl, y_tl)

    df_tl_sm = pd.concat([X_tl_sm, y_tl_sm], axis=1)
    
    return df_tl_sm


def scaling(num_vars: pd.DataFrame, scaler: str='minmax') -> pd.DataFrame:
    """
    metodo que detecta qual o melhor scaler a ser usado.

    :param df: DataFrame alvo das transformacoes
    :df type: pd.DataFrame
    :param cols: Lista de colunas que devem ser transformadas
    :cols type: list
    :return: DataFrame transformado
    :rtype: pd.Dataframe
    """

    if scaler == 'minmax':
        scaler = MinMaxScaler()
    elif scaler == 'standard':
        scaler = StandardScaler()
    elif scaler == 'normalizer':
        scaler = Normalizer()
    elif scaler == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Scaler '{scaler}' não é reconhecido. Use 'minmax', 'standard', 'normalizer', ou 'robust'.")
    
    return pd.DataFrame(scaler.fit_transform(num_vars), columns=num_vars.columns)


def one_hot_encoding(cat_vars: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """
    Executa o One Hot Encoding em cada coluna fornecida de um DataFrame.

    :param df: DataFrame alvo das transformacoes
    :df type: pd.DataFrame
    :param cols: Lista de colunas que devem ser transformadas
    :cols type: list
    :return: DataFrame transformado
    :rtype: pd.Dataframe
    """

    one_hot_encoder = OneHotEncoder(sparse_output=False)
    return pd.DataFrame(one_hot_encoder.fit_transform(cat_vars), columns=one_hot_encoder.get_feature_names_out(cat_cols))