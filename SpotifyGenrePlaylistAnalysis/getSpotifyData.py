import json
import os
import sys
import traceback

import numpy as np
import pandas as pd
import requests
import spotipy
from spotipy.oauth2 import SpotifyOAuth


class GetSpotifyData:
    def __init__(self, credentials_path):
        with open(credentials_path, 'r') as credentials_file:
            credentials_json = credentials_file.read()

        creds = json.loads(credentials_json)

        self.CLIENT_ID = creds['CLIENT_ID']
        self.CLIENT_SECRET = creds['CLIENT_SECRET']
        self.REDIRECT_URI = creds['REDIRECT_URI']
        self.USERNAME = creds['USERNAME']
        self.SCOPE = 'user-library-read playlist-read-private'

    def authenticate(self):
        try:
            self.auth = SpotifyOAuth(client_id=self.CLIENT_ID,
                                     client_secret=self.CLIENT_SECRET,
                                     redirect_uri=self.REDIRECT_URI,
                                     scope=self.SCOPE)

            self.sp = spotipy.Spotify(auth_manager=self.auth)
            self.token = self.auth.get_access_token(as_dict=False)
            print('Authentication Successful')
        except Exception:
            print('Authentication Failed')
            traceback.print_exc()

    def search_playlists(self, q, limit=20, url='https://api.spotify.com/v1/search'):
        response = requests.get(
            url,
            params={
                'q': q,
                'type': 'playlist',
                'limit': '50'
            },
            headers={
                'Authorization': f'Bearer {self.token}'
            },
        )
        return response.json()

    def spotipy_search_playlists(self, q, limit=20, offset=0):
        return self.sp.search(q=q, limit=limit, offset=offset, type='playlist')

    def unravel_dict_columns(self, df, columns):
        if type(columns) == str:
            columns = [columns]
        df = df.copy()
        for col in columns:
            sub = pd.DataFrame(list(df[col]))
            new_cols = [f'{col}_{c}' for c in sub.columns]
            sub.columns = new_cols
            df = pd.concat([df.drop(columns=col), sub], axis=1)

        return df

    def get_N_playlists(self, q, N=50):
        offset = 0
        playlists = []
        while(offset < N):
            limit = min(50, N-offset)
            try:
                res = self.spotipy_search_playlists(
                    q, limit=limit, offset=offset)
                p = res['playlists']['items']
                playlists += p          # don't append because p is a list. we want 1 list
            except Exception:
                print('Search Failed')
                traceback.print_exc()
                break
            offset += limit

        playlists_df = pd.DataFrame(playlists)
        playlists_df = playlists_df[['id', 'name', 'description', 'tracks']]
        return self.unravel_dict_columns(playlists_df, 'tracks')

    def get_playlist_tracks(self, playlist_id, offset=0, limit=100):
        url = 'https://api.spotify.com/v1/playlists/'
        response = requests.get(
            url+playlist_id,
            params={
                'offset': offset,
                'limit': limit
            },
            headers={
                'Authorization': f'Bearer {self.token}'
            },
        )
        return response.json()

    def get_all_playlists_tracks(self, playlist_id):
        total = sys.maxsize
        tracks_list = []
        offset = 0
        while offset < total:
            limit = min(100, total-offset)
            try:
                res = self.get_playlist_tracks(
                    playlist_id, offset=offset, limit=limit)
                tracks = pd.DataFrame(res['tracks']['items'])
                total = res['tracks']['total']
                tracks_list += list(tracks['track'])
            except Exception:
                print('Search Failed')
                traceback.print_exc()
                break
            offset += limit

        tracks_df = pd.DataFrame(tracks_list)
        tracks_df = self.unravel_dict_columns(tracks_df, ['artists', 'album'])
        tracks_df['playlist_id'] = playlist_id
        return tracks_df

    def get_all_tracks_from_all_playlists(self, playlist_df):
        playlist_df = playlist_df.copy()
        tracks_df = pd.DataFrame()
        for playlist_id in playlist_df['id']:
            try:
                res = self.get_all_playlists_tracks(playlist_id)
                tracks_df = pd.concat(
                    [tracks_df, res], ignore_index=True, axis=0)
            except Exception:
                print('error with playlist: ', playlist_id, 'continuing')
                traceback.print_exc()

        return tracks_df.merge(playlist_df, left_on='playlist_id', right_on='id',
                               suffixes=('_track', '_playlist'))

    # ids should be a list no more than 100 items long
    def get_batch_audio_features(self, ids):
        if type(ids) == str:
            id_string = ids
        else:
            id_string = ','.join(ids)
        url = 'https://api.spotify.com/v1/audio-features'
        response = requests.get(
            url,
            params={
                'ids': id_string
            },
            headers={
                'Authorization': f'Bearer {self.token}'
            },
        )
        if response.status_code == 200:
            return response.json()
        else:
            print('error')
            print(response.status_code)
            print(response.reason)

    def get_all_tracks_audio_features(self, df, column):
        if any(df[column].isna()):
            print(
                'There is a Null Track Id in this column, please remove it and then try again')
            return

        df = df.copy()
        ids = list(df[column])
        total = len(ids) - 1
        offset = 0
        features = []
        while(offset < total):
            print('offset:', offset)
            print('total:', total)
            window = min(100, total-offset)
            chunk = ids[offset:offset+window]
            try:
                print('getting chunk')
                res = self.get_batch_audio_features(chunk)
                print('adding features')
                features += res['audio_features']
            except Exception:
                print('Search Failed')
                traceback.print_exc()
                break
            offset += window

        features_df = pd.DataFrame(features)

        return df.merge(features_df, left_on=column, right_on='id', suffixes=('_track', '_feature'))


###############################################################
s = GetSpotifyData('credentials.json')
s.authenticate()
house = s.get_N_playlists('deep house', 100)

all_house_tracks = s.get_all_tracks_from_all_playlists(house)
house_tracks_and_playlists = all_house_tracks.merge(house, left_on='playlist_id', right_on='id',
                                                    suffixes=('_track', '_playlist'))
# s.get_N_playlists('shoegaze', 123)
# yacht = s.get_N_playlists('yacht rock', 420)

house_tracks_and_playlists = house_tracks_and_playlists[~house_tracks_and_playlists['id_track'].isna(
)]

house_features = s.get_all_tracks_audio_features(
    house_tracks_and_playlists, 'id_track')

s.get_batch_audio_features(house_tracks_and_playlists['id_track'][21600:21636])

house_tracks_and_playlists.columns

tracks = s.get_all_playlists_tracks('37i9dQZF1DX2TRYkJECvfC')
tracks.shape
tracks.columns

audio_features = s.get_batch_audio_features(house_tracks_sample)
pd.DataFrame(audio_features['audio_features'])

############################################################

offset = 21600
total = 21636
window = total-offset
chunk = house_tracks_and_playlists['id_track'][offset:offset+window]
test_features = s.get_batch_audio_features(chunk)

for i in house_tracks_and_playlists['id_track'][11100:11200]:
    print(i)

house = s.search_playlists('deep house', limit=50)
s.spotipy_search_playlists('deep house', 50, 0)

s.sp.search('deep house', type='playlist')

playlists = s.spotipy_search_playlists('deep house', 50, 0)
type(playlists['playlists']['items'])

playlist = s.sp.playlist('37i9dQZF1DX2TRYkJECvfC')
len(playlist['tracks']['items'])


yacht_joined = pd.concat([yacht, yacht_sub], axis=1)

pd.concat([yacht, yacht_sub], axis=1)


def unravel_dict_columns(df, columns):
    df = df.copy()
    for col in columns:
        sub = pd.DataFrame(list(df[col]))
        new_cols = [f'{col}_{c}' for c in sub.columns]
        sub.columns = new_cols
        df = pd.concat([df.drop(columns=col), sub], axis=1)

    return df


yacht_unraveled = unravel_dict_columns(yacht, 'tracks')
yacht_unraveled


def get_playlist_tracks(playlist_id):
    res = s.sp.playlist(playlist_id)


res = s.sp.playlist('37i9dQZF1DX2TRYkJECvfC')
res.keys()
res['tracks'].keys()
len(res['tracks']['items'])

res['tracks'].keys()
res['tracks']['total']
res['tracks']['next']


def get_playlist_tracks(playlist_id, offset=0, limit=100):
    url = 'https://api.spotify.com/v1/playlists/'
    response = requests.get(
        url+playlist_id,
        params={
            'offset': offset,
            'limit': limit
        },
        headers={
            'Authorization': f'Bearer {s.token}'
        },
    )
    return response.json()


def get_all_playlists_tracks(playlist_id, offset=0, limit=100):
    total = sys.maxsize
    tracks_list = []
    while offset < total:
        limit = min(100, total-offset)
        try:
            res = self.get_playlist_tracks(
                playlist_id, offset=offset, limit=limit)
            tracks = pd.DataFrame(res['tracks']['items'])
            tracks_list += list(tracks['track'])
        except Exception:
            print('Search Failed')
            traceback.print_exc()
            break
        offset += limit


res = get_playlist_tracks('37i9dQZF1DX2TRYkJECvfC')
res2 = get_playlist_tracks('37i9dQZF1DX2TRYkJECvfC', offset=100)
res['']
res.keys()
res['tracks'].keys()
res['tracks']['next']

res1_df = pd.DataFrame(res['tracks']['items'])
res2_df = pd.DataFrame(res2['tracks']['items'])

res1_tracks = pd.DataFrame(list(res1_df['track']))
res1_tracks.columns
s.unravel_dict_columns(res1_tracks, 'album').columns
