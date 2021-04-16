import pandas as pd
import random
import numpy
from itertools import permutations
from pandas import DataFrame
from pandas.core.frame import DataFrame
from faker import Faker
from sklearn.metrics.pairwise import cosine_similarity
fake = Faker()

# # Criando a base de dados
# Gerar úsuarios com a bibilioteca faker
def criar_email(how_many):
    email = []
    for i in range(0, how_many):
        email.append(fake.email())
    return email


email = criar_email(20)
print(email)

# Concatenar os emails
email_id = email + email
print(email_id)


# Criar função para determinar os filmes
def criar_filmes(how_many):
    titulo = ["Filme1", "Filme2", "Filme3", "Filme4", "Filme5"]
    filmes = []
    for i in range(0, how_many):
        filmes.append(random.choice(titulo))
    return filmes


filme_id = criar_filmes(40)
print(filme_id)


# Criar função para determinar a classificação
def criar_clas(how_many):
    classificacao = []
    for i in range(0, how_many):
        classificacao.append(random.randint(0, 5))
    return classificacao


classificacao_id = criar_clas(40)
print(classificacao_id)

# Criando dataframe de dados implicitos
df = pd.DataFrame(list(zip(email_id, filme_id)), columns=['Email', 'Titulo'])
print(df)
# Criando dataframe de dados explicitos
df_classificacao: DataFrame = pd.DataFrame(list(zip(email_id, filme_id, classificacao_id)),
                                columns=['Email', 'Titulo', "Classificacao"])
print(df_classificacao)

# # Recomendação de filmes mais populares

# Frequência dos filmes
Filmes_populares = df["Titulo"].value_counts()
print(Filmes_populares)

# Filtrando os filmes que aparecem mais vezes
Filmes_populares = Filmes_populares[Filmes_populares > 5].index
print(Filmes_populares)

# Usar a lista populares para filtrar os filmes mais populares no DataFrame original.
Rancking_filmes_populares = df[df["Titulo"].isin(Filmes_populares)]
print(Rancking_filmes_populares)

# Criar ranking de filmes populares
Rancking_filmes_populares.pivot_table('Email', index=['Titulo'], aggfunc="count", margins=True)

# # Sugestão baseada em outro filme
# #Os pares de filmes vistos juntos

# Criar função permutar os pares de filmes
def pares_filmes(x):
    pairs = pd.DataFrame(list(permutations(x.values, 2)),
                       columns=['FilmeA', 'FilmeB'])
    return pairs

# Ao agrupar o dataframe por título, aplicar a função para delimitar os pares
combinar_filmes = df.groupby('Email')['Titulo'].apply(pares_filmes)
print(combinar_filmes)

# Atribui na variável o número de vezes que ocorre os pares de filmes
contar_combinacao = combinar_filmes.groupby(['FilmeA', 'FilmeB']).size()

#Transformar em um Dataframe
contar_combinacao_df = contar_combinacao.to_frame(name='size').reset_index()
print(contar_combinacao_df)

#Ordenar o dataset
contar_combinacao_df.sort_values('size',ascending=False, inplace=True)
print(contar_combinacao_df.head())

# Exemplo: Sugestão  com base no filme5
Filme5 = contar_combinacao_df[contar_combinacao_df['FilmeA'] == 'Filme5']
print(Filme5)

# Visualizando os resultados
import matplotlib.pyplot as plt

Filme5.plot.bar(x="FilmeB")
plt.show()

# # Dataset com classificação
df_classificacao = df_classificacao.head(20)

# Transformar o dataset em tabela pivot,para relacionar o úsuario e filme com sua respectiva classificação
tabela_usuario_classificacao = df_classificacao.pivot(index='Email', columns='Titulo', values="Classificacao")
print(tabela_usuario_classificacao)

# Substituindo as classificações não informadas por zero.
tabela_usuario_classificacao_ = tabela_usuario_classificacao.fillna(0)

# Utilizando a transposta da matriz, para deixar a tabela baseado nos filmes
tabela_filme_classificacao_t = tabela_usuario_classificacao_.T
print(tabela_filme_classificacao_t)

# Aplicando cosine_similarity na tabela
similaridades = cosine_similarity(tabela_filme_classificacao_t)
print(similaridades)

# Transformando a tabela em um Dataframe
cosine_similarity_df = pd.DataFrame(similaridades, index=tabela_filme_classificacao_t.index,
                                    columns=tabela_filme_classificacao_t.index)

# Encontrando os filmes similares com o Filme2
cosine_similarity_series = cosine_similarity_df.loc['Filme2']

# Ordenando os valores encontrados
ordernar_similarities = cosine_similarity_series.sort_values(ascending=False)
print(ordernar_similarities)

print(ordernar_similarities)


