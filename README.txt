This is a project that uses the Spotify Web API to search for playlists and identify trends 
and associations between different tracks that come up in those playlists. This is intended 
to be used to generate new playlists for genres that are based off of what the most popular 
results for that search are, but really it could be used for any type of associating a word
with certain songs. Currently the getSpotifyData module has funtionality to search for a 
certain number of playlists for a search term or genre, get all the tracks for all those
playlists and then get all the audio features for those tracks. It can then mine for 
association rules with those tracks in the different playlists and return the set of rules
and the most popular tracks for that genre.

To get this code to work with your app, you will need to put your credentials into a 
credentials.json file in your working directory so that the code can read it and 
authenticate with Spotify to get your token to make requests.

The Jupyter Notebook shows how to use the high level methods of the module as well as
different visualizations and analyses that can be made with the results
