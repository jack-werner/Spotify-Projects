import json
import os
import sys
import traceback
from functools import reduce

import numpy as np
import pandas as pd
import requests
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import mlxtend
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


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
        ids = list(df[column].unique())
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
        # rename id column of features_df
        features_df = features_df.rename(columns={'id': 'track_id'})

        return df.merge(
            features_df,
            left_on=column,
            right_on='track_id',
            suffixes=('_track', '_playlist')
        )

    def reduce_strings(self, l, r):
        if l and r:
            # concatenate with pipe so as not to mess up csv storage
            return f"{l} | {r}"
        if not r:
            # l will never not be null since the first name/id is never null
            return l

    def fix_artists(self, df, columns, keys=['id', 'name'], track_id_col='id_track'):
        # fill artists columns nulls with empty dicts
        original = df.copy()
        df = df[columns+[track_id_col]].copy()

        res = pd.DataFrame()
        # append track id column to join back onto original df
        res[track_id_col] = df[track_id_col]

        for key in keys:
            temp = pd.DataFrame()
            for col in columns:
                # replace nulls with empty dicts so we can apply get to them
                df[col] = np.where(df[col].isnull(), {}, df[col])
                temp[f'{key}_{col}'] = df[col].apply(lambda x: x.get(key,))
            res[f"artist_{key}s"] = temp.apply(
                lambda x: reduce(self.reduce_strings, x), axis=1)
        res = res.drop_duplicates()

        return original.merge(res, on=track_id_col, how='left')

    def get_track_associations(self, df, min_sup, playlist_id='playlist_id', track_id='id_track'):
        playlists = list(df.groupby(playlist_id)[track_id].apply(list))

        print('getting frequent itemsets')
        te = TransactionEncoder()
        te_array = te.fit(playlists).transform(playlists)
        transactions_df = pd.DataFrame(te_array, columns=te.columns_)
        frequent_itemsets = apriori(
            transactions_df,
            min_support=min_sup,
            use_colnames=True,
            max_len=2)
        # return frequent_itemsets
        print('mining rules')
        rules = association_rules(
            frequent_itemsets, metric='confidence', min_threshold=0.1)

        # fix formatting for rules
        rules['antecedents'] = (rules['antecedents']
                                .astype(str)
                                .apply(lambda x: x.split("'")[1]))
        rules['consequents'] = (rules['consequents']
                                .astype(str)
                                .apply(lambda x: x.split("'")[1]))

        # get antecedent/consequent track info
        rules = rules.merge(
            df[['id_track', 'name_track',
                            'artist_names']].drop_duplicates(),
            left_on='antecedents',
            right_on='id_track')
        rules = rules.merge(
            rules[['id_track', 'name_track',
                   'artist_names']].drop_duplicates(),
            left_on='consequents',
            right_on='id_track',
            suffixes=('_antecedents', '_consequents'))

        rules.drop_duplicates()
        rules = rules.sort_values(by='confidence', ascending=False)

        # get list of all the songs that show up here
        antecedent_columns = ['antecedents', 'name_track_antecedents',
                              'artist_names_antecedents', 'antecedent support']
        consequent_columns = ['consequents', 'name_track_consequents',
                              'artist_names_consequents', 'consequent support']
        antecedents = house_rules[antecedent_columns].reset_index(drop=True)
        consequents = house_rules[consequent_columns].reset_index(drop=True)
        # rename columns
        colnames = ['track_id', 'track_name', 'artist_names', 'support']
        antecedents.columns = colnames
        consequents.columns = colnames

        tracks = pd.concat([antecedents, consequents], axis=0)
        tracks = tracks.drop_duplicates().sort_values('support', ascending=False)

        return (rules, tracks)


        ###############################################################
s = GetSpotifyData('credentials.json')
s.authenticate()

######################################################################
house = s.get_N_playlists('deep house', 100)

all_house_tracks = s.get_all_tracks_from_all_playlists(house)
house_tracks_and_playlists = all_house_tracks.merge(house, left_on='playlist_id', right_on='id',
                                                    suffixes=('_track', '_playlist'))
# s.get_N_playlists('shoegaze', 123)
# yacht = s.get_N_playlists('yacht rock', 33)
house_mask = ~house_tracks_and_playlists['id_track'].isna()
house_tracks_and_playlists = house_tracks_and_playlists[house_mask]

house_features = s.get_all_tracks_audio_features(
    house_tracks_and_playlists, 'id_track')

artist_cols = ['artists_0', 'artists_1', 'artists_2',
               'artists_3', 'artists_4', 'artists_5']
house_features = s.fix_artists(house_features, artist_cols)
######################################################################
house_features.columns
[col for col in house_features.columns if 'artists_' in col]
