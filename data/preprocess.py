import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Returns preprocessed dataframe(basic categorical changes + dealing with NaNs)
    input: pd.DataFrame
    output: pd.DataFrame
    '''
    df = df.copy()

    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

    df['Initial'] = df['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
                                            ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'])
    
    df['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)

    return df