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
            # print(offset)
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
        if response.status_code == 200:  # Can still get OK from empty response.
            return response.json()
        else:
            print('Error:')
            print(response.status_code)
            print(response.reason)
            return None

    def get_all_playlists_tracks(self, playlist_id):
        # print(playlist_id)
        total = sys.maxsize
        tracks_list = []
        offset = 0
        while offset < total:
            # print(offset)
            limit = min(100, total-offset)
            try:
                # print('fetching chunk of tracks')
                res = self.get_playlist_tracks(
                    playlist_id, offset=offset, limit=limit)
                tracks = pd.DataFrame(res['tracks']['items'])
                total = res['tracks']['total']
                if not tracks.empty:
                    # filter out Nones
                    tracks_list += list(filter(None, tracks['track']))
            except Exception:
                print('Failed to retrieve tracks.')
                traceback.print_exc()
                break
            offset += limit

        if len(tracks_list) > 0:
            tracks_df = pd.DataFrame(tracks_list)
            # print(tracks_df.shape)
            # print(tracks_df.columns)
            tracks_df = self.unravel_dict_columns(
                tracks_df, ['artists', 'album'])
            tracks_df['playlist_id'] = playlist_id
            return tracks_df
        else:
            # return empty df for compatibility with get_all_tracks_from_allplaylists
            return pd.DataFrame()

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
            # print('offset:', offset)
            # print('total:', total)
            window = min(100, total-offset)
            chunk = ids[offset:offset+window]
            try:
                # print('getting chunk')
                res = self.get_batch_audio_features(chunk)
                # print('adding features')
                features += res['audio_features']
            except Exception:
                print('Search Failed')
                traceback.print_exc()
                break
            offset += window
        # print('about to make df')
        # filter Nones from features_list
        features = list(filter(None, features))
        features_df = pd.DataFrame(features)
        # print('df creation successful')

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
playlist_id = '6ZSE0F0CEWHDAHOVZjrZVh'
s.get_all_playlists_tracks(playlist_id)

tracks = s.get_all_playlists_tracks(playlist_id)

df = pd.DataFrame(tracks)
df.head()

df = pd.DataFrame(tracks[2:])

len(tracks)

l = [1 if t == None else 0 for t in tracks]
sum(l)
