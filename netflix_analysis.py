"""Importing necessary Packages"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

"""Loading the Dataset"""
netflix_titles_df = pd.read_csv(r'C:\Users\SAKSHAM\Desktop\Python Codes\Netflix_Analysis\netflix_titles.csv')
print(netflix_titles_df.head())

"""By looking at the dataset, 
it looks like the content in the 
dataset is without user ratings. 
Also see there are NaN values in 
some columns.
So let's get started""" 

#DATA PREPARATION AND CLEANING
print(netflix_titles_df.info())

"""As per the output there are 6234 shows
and 12 columns.These columns also includes
NUll values for 'director','cast','country',
'date added','ratings'."""

#Handling NULL Values
print(netflix_titles_df.isnull().values.any())
print(netflix_titles_df.isnull().sum().sum())             #To find the exact number of Null values

sns.heatmap(netflix_titles_df.isnull(),cbar=False)
plt.title('Null Values Heatmap')
plt.show()
print(netflix_titles_df.isnull().sum())

"""
With the assistance of heatmap and table, we can see that there are quite a few null values in the dataset. 
There are a total of 3,036 null values across the 
entire dataset with 1,969 missing points 
under 'director', 570 under 'cast', 
476 under 'country', 11 under 
'date_added', and 10 under 'rating'. 
We will have to handle all null data 
points before we can dive into EDA and 
modeling.
"""
netflix_titles_df['director'].fillna('No Director', inplace=True)
netflix_titles_df['cast'].fillna('No Cast', inplace=True)
netflix_titles_df['country'].fillna('Country Unavailable', inplace=True)
netflix_titles_df.dropna(subset=['date_added','rating'],inplace=True)

print(netflix_titles_df.isnull().any())
"""
For null values, the easiest way is to delete the rows 
with the missing data. However, this wouldn't be beneficial to 
our EDA since there is loss of information. 
Since 'director', 'cast', and 'country' contain 
the majority of null values, Instead of this each 
missing value will be treated as unavailable. The other two labels 
'date_added' and 'rating' contains an insignificant portion 
of the data so let's drop them from the dataset. 
After, we can see that there are no more null values in the 
dataset.

"""
#Splitting the Dataset into Movies and Web Series
netflix_movies_df = netflix_titles_df[netflix_titles_df['type']=='Movie'].copy()
print(netflix_movies_df.head())

netflix_shows_df = netflix_titles_df[netflix_titles_df['type']=='TV Show'].copy()
print(netflix_shows_df.head())

#Data Preparation
"""
In the duration column, there appears to be a discrepancy between movies and shows. 
Movies are based on the duration of the movie and shows are based on the number of seasons. 
To make EDA easier, I will convert the values in these columns into integers for both the movies 
and shows datasets.
"""
netflix_movies_df.duration = netflix_movies_df.duration.str.replace(' min','').astype(int)
netflix_shows_df.rename(columns={'duration':'seasons'}, inplace=True)
netflix_shows_df.replace({'seasons':{'1 Season':'1 Seasons'}}, inplace=True)
netflix_shows_df.seasons = netflix_shows_df.seasons.str.replace(' Seasons','').astype(int)

# Exploratory Analysis and Visualization

"""
First we will begin analysis on the entire Netflix dataset consisting of both movies and shows. 
Revisiting the data, let us see how it looked like again.
"""
print(netflix_titles_df.head())

"""
Netflix Film Types: Movie or TV Show
It'd be interesting to see the comparison between the total number of movies and shows in this 
dataset just to get an idea of which one is the majority.
"""
plt.figure(figsize=(7,5))
g = sns.countplot(netflix_titles_df.type, palette="pastel");
plt.title("Count of Movies and TV Shows")
plt.xlabel("Type (Movie/TV Show)")
plt.ylabel("Total Count")
plt.show()

plt.figure(figsize=(12,6))
plt.title("% of Netflix Titles that are either Movies or TV Shows")
g = plt.pie(netflix_titles_df.type.value_counts(), explode=(0.025,0.025), labels=netflix_titles_df.type.value_counts().index, colors=['skyblue','navajowhite'],autopct='%1.1f%%', startangle=180);
plt.legend()
plt.show()

"""
So there are roughly 4,000+ movies and almost 2,000 shows with movies 
being the majority. This makes sense since shows are always an ongoing 
thing and have episodes. If we were to do a headcount of TV show episodes vs. 
movies, I am sure that TV shows would come out as the majority. 
However, in terms of title, there are far more movie titles (68.5%) than 
TV show titles (31.5%).

#Netflix Film Ratings

Now, we will explore the ratings which are based on the film rating system. 
The ordering of the ratings will be based on the
age of the respective audience from youngest to oldest. We will not include 
the ratings 'NR' and 'UR' in the visuals since they stand for unrated and non-rated 
content.
"""
order =  ['G', 'TV-Y', 'TV-G', 'PG', 'TV-Y7', 'TV-Y7-FV', 'TV-PG', 'PG-13', 'TV-14', 'R', 'NC-17', 'TV-MA']
plt.figure(figsize=(15,7))
g = sns.countplot(netflix_titles_df.rating, hue=netflix_titles_df.type, order=order, palette="pastel");
plt.title("Ratings for Movies & TV Shows")
plt.xlabel("Rating")
plt.ylabel("Total Count")
plt.show()

fig, ax = plt.subplots(1,2, figsize=(19, 5))
g1 = sns.countplot(netflix_movies_df.rating, order=order,palette="Set2", ax=ax[0]);
g1.set_title("Ratings for Movies")
g1.set_xlabel("Rating")
g1.set_ylabel("Total Count")
g2 = sns.countplot(netflix_shows_df.rating, order=order,palette="Set2", ax=ax[1]);
g2.set(yticks=np.arange(0,1600,200))
g2.set_title("Ratings for TV Shows")
g2.set_xlabel("Rating")
g2.set_ylabel("Total Count")
fig.show()

"""
Overall, there is much more content for a more mature audience. For the mature audience, 
there is much more movie content than there are TV shows. However, for the younger 
audience (under the age of 17), it is the opposite, there are slightly more TV shows than 
there are movies.
"""
netflix_titles_df['year_added'] = pd.DatetimeIndex(netflix_titles_df['date_added']).year
netflix_movies_df['year_added'] = pd.DatetimeIndex(netflix_movies_df['date_added']).year
netflix_shows_df['year_added'] = pd.DatetimeIndex(netflix_shows_df['date_added']).year
netflix_titles_df['month_added'] = pd.DatetimeIndex(netflix_titles_df['date_added']).month
netflix_movies_df['month_added'] = pd.DatetimeIndex(netflix_movies_df['date_added']).month
netflix_shows_df['month_added'] = pd.DatetimeIndex(netflix_shows_df['date_added']).month

"""
#Content added each year

Now we will take a look at the amount content Netflix has added throughout the previous years. 
Since we are interested in when Netflix added the title onto their platform, we will add a 
'year_added' column shows the year of the date from the 'date_added' column as shown above.
"""
netflix_year = netflix_titles_df['year_added'].value_counts().to_frame().reset_index().rename(columns={'index': 'year','year_added':'count'})
netflix_year = netflix_year[netflix_year.year != 2020]
print(netflix_year)

netflix_year2 = netflix_titles_df[['type','year_added']]
movie_year = netflix_year2[netflix_year2['type']=='Movie'].year_added.value_counts().to_frame().reset_index().rename(columns={'index': 'year','year_added':'count'})
movie_year = movie_year[movie_year.year != 2020]
show_year = netflix_year2[netflix_year2['type']=='TV Show'].year_added.value_counts().to_frame().reset_index().rename(columns={'index': 'year','year_added':'count'})
show_year = show_year[show_year.year != 2020]

fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=netflix_year, x='year', y='count')
sns.lineplot(data=movie_year, x='year', y='count')
sns.lineplot(data=show_year, x='year', y='count')
ax.set_xticks(np.arange(2008, 2020, 1))
plt.title("Total content added each year (up to 2019)")
plt.legend(['Total','Movie','TV Show'])
plt.ylabel("Releases")
plt.xlabel("Year")
plt.show()

"""
Based on the above timeline, we can see that the popular 
streaming platform started gaining traction after 2014. 
Since then, the amount of content added has been tremendous. 
I decided to exclude content added during 2020 since the data 
does not include a full years worth of data. We can see that 
there has been a consistent growth in the number of movies on 
Netflix compared to shows.
"""
month_year_df = netflix_titles_df.groupby('year_added')['month_added'].value_counts().unstack().fillna(0).T

plt.figure(figsize=(11,8))
sns.heatmap(month_year_df, linewidths=0.025, cmap="YlGnBu")
plt.title("Content Heatmap")
plt.ylabel("Month")
plt.xlabel("Year")
plt.show()

"""
In the above heatmap, 
we can see that around 2014 is when 
Netflix began to increase their content 
count. We can see over the years and months, 
Netflix continues to slowly increase the amount 
of content that is being added into their platform. 
We can see in 2020, the data stops at January since 
that is the latest month available in the dataset.
"""
#Netflix Film Duration
fig, ax = plt.subplots(1,2, figsize=(19, 5))
g1 = sns.distplot(netflix_movies_df.duration, color='skyblue',ax=ax[0]);
g1.set_xticks(np.arange(0,360,30))
g1.set_title("Duration Distribution for Netflix Movies")
g1.set_ylabel("% of All Netflix Movies")
g1.set_xlabel("Duration (minutes)")
g2 = sns.countplot(netflix_shows_df.seasons, color='skyblue',ax=ax[1]);
g2.set_title("Netflix TV Shows Seasons")
g2.set_ylabel("Count")
g2.set_xlabel("Season(s)")
fig.show()

"""
Now we will look into the duration of Netflix films. Since movies are measured in 
time and shows are measured by seasons, we need to split the dataset between movies 
and TV shows. Above on the left, we can see that the duration for Netflix movies closely 
resembles a normal distribution with the average viewing time spanning about 90 minutes which seems to make sense. Netflix TV shows on the other hand seems to be heavily skewed to the right where the majority of shows only have 1 season.
"""

#Countries with the most content available
filtered_countries = netflix_titles_df.set_index('title').country.str.split(', ', expand=True).stack().reset_index(level=1, drop=True);
filtered_countries = filtered_countries[filtered_countries != 'Country Unavailable']

plt.figure(figsize=(7,9))
g = sns.countplot(y = filtered_countries, order=filtered_countries.value_counts().index[:20])
plt.title('Top 20 Countries on Netflix')
plt.xlabel('Titles')
plt.ylabel('Country')
plt.show()

"""
Now we will explore the countries with the most content on Netflix. Films typically are available 
in multiple countries as shown in the original dataset. Therefore, we need to seperate all countries 
within a film before we can analyze the data. After seperating countries and removing titles with no 
countries available, we can plot a Top 20 list to see which countries have the highest availability 
of films on Netflix.

Unsurprisingly, the United States stands out on top since Netflix is an American company. India surprisingly 
comes in second followed by the UK and Canada. China interestingly is not even close to the top even though 
it has about 18% of the world's population. Reasons for this could be for political reasons and the banning of 
certain applications which isn't uncommon between the United States and China.
"""

#Popular Genres
filtered_genres = netflix_titles_df.set_index('title').listed_in.str.split(', ', expand=True).stack().reset_index(level=1, drop=True);

plt.figure(figsize=(7,9))
g = sns.countplot(y = filtered_genres, order=filtered_genres.value_counts().index[:20])
plt.title('Top 20 Genres on Netflix')
plt.xlabel('Titles')
plt.ylabel('Genres')
plt.show()
"""
In terms of genres, international movies takes the cake surprisingly followed by dramas and comedies.
Even though the United States has the most content available, it looks like Netflix has decided to 
release a ton of international movies. 
The reason for this could be that most 
Netflix subscribers aren't actually in the United States, but rather the majority 
of viewers are actually international subscribers."""