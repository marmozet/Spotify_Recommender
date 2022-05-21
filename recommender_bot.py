import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from yellowbrick.target import FeatureCorrelation
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
import json
import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv("data/data_tracks.csv")


number_cols = ['valence', 'danceability', 'energy', 'key', 'loudness','popularity','year']

#number_cols =['acousticness', 'danceability', 'energy', 'instrumentalness',
#            'liveness', 'speechiness', 'tempo', 'valence','duration_ms','key','year']

#number_cols = ['danceability','popularity','year']
n_c = 4

# mlflow.sklearn.autolog()
params = {"n_clusters": n_c}

#Generamos un flujo de trabajo con dos procesos
song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), 
                                  ('kmeans', KMeans(**(params), 
                                   verbose=False))
                                 ], verbose=False)

#pasamos las columnas que utilizaremos para el entrenamiento
X = data[number_cols]
song_cluster_pipeline.fit(X)
#Predecimos el grupo que pertenece cada canción
song_cluster_labels = song_cluster_pipeline.predict(X)
#Asignamos los labels del resultado de nuestro entrenamiento a nuestro dataframe
data['cluster_label'] = song_cluster_labels


sse = []
list_k = list(range(2, n_c))
scaler = song_cluster_pipeline.steps[0][1]
scaled_data = scaler.fit_transform(data[number_cols])



from sklearn.decomposition import PCA

# Visualizing the Clusters with PCA

#Generamos un flujo de trabajo con dos procesos
pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
#Vectorizamos las caracteristicas para encontrar los componentes principales
embedding = pca_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=embedding)
projection['id'] = data['id']
projection['title'] = data['name']
projection['cluster'] = data['cluster_label']




length = len(number_cols)
for i in range(length):
    projection[number_cols[i]] = projection.id.map(data.set_index('id')[number_cols[i]])
    
df = projection.sort_values('cluster')
for i in range(n_c):
    globals()['cluster%s' % i]=df.loc[df.loc[:, 'cluster'] == int(i)]





# Insert spotify api tokens here

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="Insert client_id here",
                                                           client_secret="Insert client_secret here"))

s = pd.DataFrame()
def find_song(name):
    song_data = defaultdict()
    results = sp.search(q= 'track: {}'.format(name), limit=1)
    if results['tracks']['items'] == []:
        return None
    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]
    song_data['name'] = [name]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['popularity'] = [results['popularity']]
    list_of_dict_values = list(results.values())
    results_string = json.dumps(list_of_dict_values)
    index_release_date = results_string.find("release_date")
    year_index1 = index_release_date + 16
    year_index2 = index_release_date + 17
    year_index3 = index_release_date + 18
    year_index4 = index_release_date + 19
    year_string = results_string[year_index1] + results_string[year_index2] + results_string[year_index3] + results_string[year_index4]
    print(year_string)
    
    song_data['year'] = [int(year_string)]
    for key, value in audio_features.items():
        song_data[key] = value
    s=pd.DataFrame(song_data)
    return s






def get_song_data(song, spotify_data):
    
    return find_song(song['name'])
        

def get_mean_vector(song_list, spotify_data):
    
    song_vectors = []
    
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)  
    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)


def list_dict(dict_list):
    
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
            
    return flattened_dict


def recommend_songs( song_list, spotify_data, n_songs=3):
    metadata_cols = ['id','name','year','artists']
    song_dict = list_dict(song_list)
    song_center = get_mean_vector(song_list, spotify_data)
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    song_cluster_label = song_cluster_pipeline.predict(scaled_song_center)
    s['cluster_label'] = song_cluster_label
    cluster_s=df.loc[df.loc[:, 'cluster'] == int(s['cluster_label'])]
    cluster_scaled = scaler.fit_transform(cluster_s[number_cols])
    distances = cdist(scaled_song_center, cluster_scaled, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])

    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols]








import discord

# Insert your token here
TOKEN = "Insert discord bot token here"



url_1 = "https://open.spotify.com/track/"

client = discord.Client()


@client.event
async def on_ready():
    print("{0.user} is online!".format(client))


@client.event
async def on_message(message):

    message1 = message.content

    # list_a = ['!rec','Breaking All The Rules']

    split_message = message1.split(None, 1)
    split_message_1 = split_message[1]
    str_split_message_1 = str(split_message_1)


    # Se pone en el formato de diccionario que se usa como input
    string_dict = "{'name': " + "'" + str_split_message_1 + "'" + "}"


    # Ponemos todo en formato necesario del input
    dict_test = eval(string_dict)
    input_song = [dict_test]



    df_recommended_songs = recommend_songs(input_song,  data)

    df_recommended_songs.reset_index(drop=True, inplace=True)
    df_recommended_songs


    id_list = df_recommended_songs['id'].tolist()
    print(id_list)


    # print(split_message)

    if message.author == client.user:
        return

    elif split_message[0] == "!rec":

        if len(split_message) < 1:
            await message.channel.send("Song name is needed")
        else:
            await message.channel.send(url_1 + id_list[0])
            await message.channel.send(url_1 + id_list[1])
            await message.channel.send(url_1 + id_list[2])


client.run(TOKEN)

