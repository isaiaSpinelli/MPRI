# This is a sample Python script.

# https://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/#loc-selection

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    a = np.arange(12).reshape(3, 4)
    print(a)
    print(a.shape)
    # x,y
    a[1,2] = 0
    print(a)

    # print 1er row
    print(a[0, : ])
    # print 2eme column
    print(a[ : ,1])

    b = a[ :2, :2]
    print(b)
    b[0,0] = 9
    print(b)
    print(a)
    # les elements de b sont les mêmes que ceux de a ...

    print(a.size)
    print(a[0, : ].size)
    print(a[ : ,0].size)

    print("\n\n\n")


    # a Pandas series est une série d'un type ( plusieurs noms d'auteurs)
    # une dataFrame : est composée de plusieurs series.

    #  NumPy ndarray : Tableau multi dimensionnels
    #  Pandas dataframes : Tableau 2D  avec des labels (pouvant contenir plusieurs type différent par colonne)
    import pandas as pd

    author = ['Jitender', 'Purnima', 'Arpit', 'Jyoti']

    auth_series = pd.Series(author)
    print(type(auth_series))

    # Dictionary of string and int
    char_dict = {
        'C': 56,
        "A": 23,
        'D': 43,
        'E': 78,
        'B': 11
    }
    # Convert a dictionary to a Pandas Series object.
    # dict keys will be index of Series &
    # values of dict will become values in Series.
    series_obj = pd.Series(char_dict)
    print('Contents of Pandas Series: ')
    print(series_obj)

    # DATAFRAMES
    data = pd.read_csv("us-500.csv")
    print(data)

    Row1 =data.iloc[0]
    print(Row1)
    print(type(Row1))

    Row1DF = data.iloc[[0]]
    print(Row1DF)
    print(type(Row1DF))

    LastRow = data.iloc[-1:,:]
    print(LastRow)
    print(type(LastRow))

    firstColomn = data.iloc[:,0]
    print(firstColomn)

    Colomn2 = data.iloc[:, 1]
    print(Colomn2)

    lastcolomn = data.iloc[:,-1:]
    print(lastcolomn)

    firstFiveRows = data.iloc[:5,:]
    print(firstFiveRows)

    secondColOfAllRow = data.iloc[:,:2]
    print(secondColOfAllRow)

    indexes = data.iloc[ [1,4,7,25] , [1,6,7]]
    print(indexes)

    indexes_2 = data.iloc[ :5, [5, 6, 7]]
    print(indexes_2)

    # Oblige de toujours voir cette colonne !
    #data.set_index('last_name', 1, 0, True)
    data.set_index("last_name", inplace=True)
    print(data)


    print(data.loc[['Rim', 'Perin'], ['first_name', 'address', 'email']])

    # Erreur ..  Select rows with index values 'Antonio' and 'Veness', with all columns between 'city' and 'email'
    # 2.2.3 (c -> d )
    #print(data.loc[('Antonio', 'Veness'), 'city'])

    print(data.loc[data['first_name'] == 'Erick'])

    # Series
    print(data.loc[data['first_name'] == 'Erick', 'email'])
    #Dataframe
    print(data.loc[data['first_name'] == 'Erick', ['email']])

    print(data.loc[data['first_name'] == 'Erick'].iloc[:,3:10] )


    print(data.loc[data['email'] == 'hotmail.com'])

    print(data.loc[data['email'].str.endswith("hotmail.com")]   )

    print(data.loc[data['company_name'].apply(lambda x: len(x.split(' ')) == 4)   ])