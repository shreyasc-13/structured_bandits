import pandas as pd
from collections import defaultdict
import numpy as np
import pickle
import argparse
import os
import sys
import random
import pathlib

def parse_arguments():
    # Command line flags to determine train-test ratio
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_test_ratio', dest='train_test_ratio', type=float, default=0.5,
                        help="Train test split ratio")

    return parser.parse_args()

def movielens_preproc():
    '''
    Add metadata and store it as a dataframe
    '''
    # Add headers
    ratings_cols = 'UserID::MovieID::Rating::Timestamp'.split('::')
    movies_cols = 'MovieID::Title::Genres'.split('::')
    users_cols = 'UserID::Gender::Age::Occupation::Zip-code'.split('::')
    # Read data
    ratings = pd.read_csv('../data/ml-1m/ratings.dat', sep='::', engine='python', names=ratings_cols)
    movies = pd.read_csv('../data/ml-1m/movies.dat', sep='::', engine='python', names=movies_cols)
    users = pd.read_csv('../data/ml-1m/users.dat', sep='::', engine='python', names=users_cols)

    # Randomly assign one genre out of multiple to each movie
    gs = movies['Genres'].tolist()
    new_genre_list = []
    for i in range(len(movies)):
        t = gs[i].split('|')
        random.shuffle(t)
        new_genre_list += [t[0]]

    movies = movies.assign(Genre_Assigned=new_genre_list)

    # Merge dataframes
    temp = pd.merge(ratings, users, how='left', on='UserID')
    data = pd.merge(temp, movies, how='left', on='MovieID')

    # Each meta-user is identified by the age group and occupation
    # Below categories as taken from: http://files.grouplens.org/datasets/movielens/ml-1m-README.txt
    ages = [1, 18, 25, 35, 45, 50, 56]

    occupations = list(range(21))

    meta_users = [(x, y) for x in ages for y in occupations]

    # Adding columns for meta-user and genre-ids for easier lookup
    genre_order = ['Mystery', 'Drama', 'Sci-Fi', "Children's", 'Horror', 'Film-Noir', 'Crime', 'Romance',
                   'Fantasy', 'Musical', 'Animation', 'Adventure', 'Action', 'Comedy', 'Documentary', 'War',
                   'Thriller', 'Western']
    # IDs for genres
    genre_ids = dict(zip(genre_order, list(range(18))))

    age = data['Age'].tolist()
    occ = data['Occupation'].tolist()
    genr = data['Genre_Assigned'].tolist()

    # IDs for meta-users
    meta_user_ids = dict(zip(meta_users, list(range(147))))

    meta_user_col = []
    genre_id_col = []
    for i in range(len(data)):
        meta_user_col += [meta_user_ids[(age[i], occ[i])]]
        genre_id_col += [genre_ids[genr[i]]]

    data = data.assign(Meta_User_Col=meta_user_col)
    data = data.assign(Genre_Col=genre_id_col)

    pathlib.Path('genres').mkdir(parents=False, exist_ok=True)
    data.to_pickle('genres/data_with_id')

    print(f"Pre-processing complete. Processed data dumped at {os.getcwd() + '/genres/data_with_id'}")

def _process_data(data, meta_users):
    meta_user_size = []
    genre_rating_users = []
    rating_count = []
    for user in meta_users:
        rating_count_user = {}
        for g in genres:
            rating_count_user[g] = 0.
        user_genres = data[(data.Age == user[0]) & (data.Occupation == user[1])]['Genre_Assigned'].tolist()
        for i in range(len(user_genres)):
            if '|' in user_genres[i]:
                user_genres[i] = user_genres[i].split('|')
            else:
                user_genres[i] = [user_genres[i]]

        meta_user_size += [len(user_genres)]

        user_ratings = data[(data.Age == user[0]) & (data.Occupation == user[1])]['Rating'].tolist()

        genre_rating = {}
        for g in genres:
            genre_rating[g] = 0.

        for i in range(len(user_genres)):
            for g in user_genres[i]:
                genre_rating[g] += user_ratings[i]
                rating_count_user[g] += 1.

        genre_rating_users += [genre_rating]
        rating_count += [rating_count_user]

    full_data_rating = []
    for i in range(len(meta_user_size)):
        temp = {}
        for g in genres:
            if rating_count[i][g] != 0:
                temp[g] = genre_rating_users[i][g]/rating_count[i][g]
            else:
                temp[g] = 0.
        full_data_rating += [temp]

    return meta_user_size, full_data_rating


def genre_train_test_split(train_test_ratio=0.5):
    genres = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama',
              'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
              'War', 'Western']
    ages = [1, 18, 25, 35, 45, 50, 56]
    occupations = list(range(21))
    meta_users = [(x, y) for x in ages for y in occupations]

    data = pd.read_pickle('genres/data_with_id')

    meta_user_size, _ = _process_data(data, meta_users)

    # Split train and test at 'train_test_ratio'
    # TODO - update to cleaner code
    test = []
    train = []

    for i in range(len(meta_user_size)):
        user = meta_users[i]
        size = meta_user_size[i]
        temp = data[(data.Age == user[0]) & (data.Occupation == user[1])]
        idx = np.arange(size)
        np.random.shuffle(idx)
        train_indices = idx[: int(0.5*size)]
        test_indices = idx[int(0.5*size) :]
        test.append(temp.iloc[test_indices])
        train.append(temp.iloc[train_indices])

    test_data = pd.concat(test)
    train_data = pd.concat(train)

    train_data.to_pickle('genres/train_data_with_id')
    test_data.to_pickle('genres/test_data_with_id')

    print(f"Train-test split complete with a train-test ratio of: {train_test_ratio}")


def genre_data_process():
    ages = [1, 18, 25, 35, 45, 50, 56]
    occupations = list(range(21))
    meta_users = [(x, y) for x in ages for y in occupations]

    train_data_id = pd.read_pickle('genres/train_data_with_id')
    test_data_id = pd.read_pickle('genres/test_data_with_id')

    train_meta_user_size, train_data_rating = _process_data(train_data, meta_users)
    train_true_X = np.array(train_meta_user_size)/float(sum(train_meta_user_size))

    test_meta_user_size, test_data_rating = _process_data(test_data, meta_users)
    test_true_X = np.array(test_meta_user_size)/float(sum(test_meta_user_size))

    np.save('genres/test_true_X', test_true_X)
    np.save('genres/test_data_rating', test_data_rating)
    np.save('genres/train_data_rating', train_data_rating)

def main(args):
    args = parse_arguments()
    movielens_preproc()
    # genre exps
    genre_train_test_split(args.train_test_ratio)
    genre_data_process()


if __name__ == '__main__':
    main(sys.argv)
