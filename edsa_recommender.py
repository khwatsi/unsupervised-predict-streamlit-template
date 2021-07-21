"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import base64
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration


def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = [
        "Recommender System",
        "Data Exploration",
        "Recommenders",
        "Developers"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png', use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option', title_list[14930:15200])
        movie_2 = st.selectbox('Second Option', title_list[25055:25255])
        movie_3 = st.selectbox('Third Option', title_list[21100:21200])
        fav_movies = [movie_1, movie_2, movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(
                            movie_list=fav_movies, top_n=10)
                    st.title("We think you'll like:")
                    for i, j in enumerate(top_recommendations):
                        st.subheader(str(i + 1) + '. ' + j)
                except BaseException:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")

        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(
                            movie_list=fav_movies, top_n=10)
                    st.title("We think you'll like:")
                    for i, j in enumerate(top_recommendations):
                        st.subheader(str(i + 1) + '. ' + j)
                except BaseException:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")

    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Recommenders":
        st.title("Recommenders")
        st.info("Recommender systems are among the most popular applications of data science today. They are used to predict the rating or preference that a user would give to an item. Almost every major tech company has applied them in some form. **Amazon** uses it to suggest products to customers, **YouTube** uses it to decide which video to play next on autoplay, and **Facebook** uses it to recommend pages to like and people to follow.")
        solution = st.selectbox(
            "We have 2 models which we use to make recommendations, which one would you like to learn about?",
            ('Select Option',
             'Content-Based Recommender',
             'Collaborative Recommender'))

        # Content-Based Recommender
        if solution == "Content-Based Recommender":
            st.write("The idea behind Content-based (cognitive filtering) recommendation system is to recommend an item based on a comparison between the content of the items (i.e. movies) and a user profile. In simple words, this system suggest similar items (movies) based on the description of a particular item (movie). This system uses item metadata, such as genre, director, description, actors, etc. For movies, to make these recommendations. The general idea behind these recommender systems is that if a person likes a particular item, he or she will also like an item that is similar to it. And to recommend that, it will make use of the user's past item metadata.")
            st.image("resources/imgs/content-based.png")
            st.write("Our Content-Based Recommender makes movie recommendations based on the **metadata (genre, cast, director and plot keywords)** of a previously watched movie.")
            st.write("It then uses a **CountVectorizer** to build numeric features from the metadata. Numeric features can be built from text data either using a **CountVectorizer** or a **TfIdf Vectorizer**, we decided to use the **CountVectorizer** because there might be many movies with the same director or actor and we definitely don't wanna penalize those directors or actors. It might be possible that a user wants to be recommended movies based on a favourite director or actor.")
            st.image("resources/imgs/countvectorizer.png")
            #st.write("Source: https://learning.oreilly.com/library/view/applied-text-analysis/9781491963036/ch04.html")
            # Similarity Index
            st.write("To then make recommendations our Content-Based Recommender calculates the similarity between the movies previously watched by a user and all the available movies in our app. This similarity is calculated using the **Cosine Similarity**, movies are then recommended based on the similarity score or 'how similar' they are to the previously watched movies.")
            st.image("resources/imgs/cosine.png")
            #st.write("Source: https://heartbeat.fritz.ai/recommender-systems-with-python-part-i-content-based-filtering-5df4940bd831")

            # Ending
            st.write("And in three simple steps, the Content-Based Recommender gives recommendations to the user on which movies to watch based on previously seen movies.")

        # Collaborative Recommender
        if solution == "Collaborative Recommender":
            st.write("Collaborative filtering methods for recommender systems are methods that are based solely on the past interactions recorded between users and items in order to produce new recommendations. These interactions are stored in the so-called **user-item interactions matrix**.These systems are widely used, and they try to predict the rating or preference that a user would give an item (movies) based on past ratings and preferences of other users. Collaborative filters do not require item metadata (genre, directors, actors, ... etc) like its content-based counterparts. It works by searching a large group of people and finding a smaller set of users with tastes similar to a particular user. It looks at the items they like and combines them to create a ranked list of suggestions.")
            st.image("resources/imgs/collaborative.png")
            st.info(
                "The class of collaborative filtering algorithms  can be divided into two sub-categories:")
            st.write("**Memory based**: Memory based approaches directly works with values of recorded interactions, assuming no model, and are essentially based on nearest neighbours search (for example, find the closest users from a user of interest and suggest the most popular items among these neighbours).")
            st.write("**Model based**: Model based approaches assume an underlying 'generative' model that explains the user-item interactions and try to discover it in order to make new predictions.")
            st.write(
                "Our Collaborative Recommender is a model-based recommender, using the SVD model from the Surprise package.")
            st.write("In the SVD (Singular Value Decomposition) method, the sparse user-movie (ratings) matrix is compressed into a dense matrix by applying matrix factorization techniques. If M is a user* movie matrix, SVD decomposes it into 3 parts: M = UZV, where U is user concept matrix, Z is weights of different concepts and V is concept movie matrix. ‘Concept’ can be intuitively understood by imagining it as a superset of similar movies like a ‘suspense thriller’ genre can be a concept, etc.")
            st.write("Once SVD decomposes the original matrix into 3, the dense matrix is directly used for predicting the rating for a (user, movie) pair using the concept to which the input_movie belongs.")
            st.image("resources/imgs/Matrix.png")
            # st.write("https://towardsdatascience.com/introduction-to-recommender-systems-6c66cf15ada")
            st.write("Using a combination of functions, our Collaborative Recommender gives recommendations to the user on which movies to watch based on previously seen movies.")

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
    if page_selection == "Data Exploration":
        st.title("Data Exploration")
        st.info("This page is dedicated to the exploratory data analysis of the data, which aided in a better understanding of the\
        problem statement. The various datasets were blended to obtain more information and highlight the underlying patterns within the data.")
        option = st.selectbox(
            "How would you like to analyse and view the data?",
            ('Select Option',
             'Ratings',
             'Movies',
             'Genres',
             'Actors & Directors'))

        # Ratings section of the EDA
        if option == 'Ratings':
            st.image("resources/imgs/ratings per year.png")

            st.write("Since the early 2000s, we've seen an upsurge in the number of movie ratings. This could be attributed to increased internet availability and use, as well as the creation of movie rating services like **IMDb, Metacritic, and RottenTomatoes**.")

            st.image("resources/imgs/rating percentage.png")
            st.write(
                "Approximately 80% of the movies have a rating of 3 or higher, making this is a collection of enjoyable movies.")
            st.image("resources/imgs/average rating per genre.png")
            st.write("Among the genres, **Film-Noir** has the highest average rating, while **Horror** has the lowest average rating. According to this article regarding the most-seen movie genres on Netflix, https://www.whats-on-netflix.com/news/what-movie-tv-genres-perform-well-in-the-netflix-top-10s/, **Horror** is only watched by roughly 4.2 percent of Netflix customers.")
            # Most popular users
            st.image("resources/imgs/Top10  most popular users.png")
            st.write("More than 12000 movies have been rated by User **72315**. This user may have begun rating movies before the majority of other users, or he or she may simply be a dedicated movie watcher.")
            # Most popular users and their ratings
            st.image("resources/imgs/Top10 most popular users and rating.png")
            st.write(
                "With the most ratings, user **72315** has an average rating of 3.")

        # The movies section of the EDA
        if option == "Movies":
            st.image("resources/imgs/number of movies per year.png")
            st.write("There is a general trend here, with the number of movies released in the twenty-first century steadily increasing (2000 - present). Since the early 2000s, the number of movie-watching options has grown, including cinemas and streaming services. Only the years in which more than 500 movies were released are included in the graph.")
            st.image("resources/imgs/highest budget movies.png")
            st.write(
                "**My Way (Mai Wei)** had a $300 million budget, whereas **Fateless (Sorstalansag)** had a $30 million budget.")
            st.image("resources/imgs/Total movie per genre.png")
            st.write("**Drama** dominates the dataset with around 4.5 million movies, followed by **Comedy** with approximately 3.6 million movies. The genres **Documentary** and **Film-Noir** have the fewest movies in the dataset. In the **Ratings** analysis, **Film-Noir** was the highest-rated genre, however, it now seems like it's because it has the fewest movies in the dataset, implying that there was a bias.")
            st.image("resources/imgs/frequent words in movie titles.png")
            st.write("**Man, Love, Girl** appears to be the most commonly used words in movie titles. **Drama** is the most common genre, and **Drama** movies are often about **Love**.")
            st.image("resources/imgs/popular movies.png")
            st.write("The most popular movies are **The Shawshank Redemption, Forrest Gump, Pulp Fiction, The Silence of the Lambs, and The Matrix**. Some believe these movies to be among the best movies of all time.")

        # Genres Section of the EDA
        if option == "Genres":
            st.image("resources/imgs/genres.png")
            st.write("**Drama** and **Comedy** are the most popular genres, with **Film-Noir** and **IMAX** being the least popular. **Drama** is the cheapest genre to produce as movies don’t necessarily require special sets, costumes, locations, props, special/visual effects, etc. **IMAX** movies on the other hand are usually displayed in theatres with special screens at times the screen sizes are six times larger than regular theatre screens.")

        # Cast/Directors section of the EDA
        if option == "Actors & Directors":
            st.image("resources/imgs/Actors2.png")
            st.write("**Tom Hanks** is an actor and director from the United States. Hanks is a prominent and recognizable film performer who is widely considered an American cultural icon. He is known for both humorous and tragic roles.")
            st.write("**Ben Stiller** has written, appeared in, directed, or produced over 50 films throughout his career, including The Secret Life of Walter Mitty, Zoolander, The Cable Guy, and There's Something About Mary.")
            st.write("Both **Eddie Murphy** and **Chris Rock** are successful comedians who have starred in several films throughout their careers.")

            st.image("resources/imgs/moviedirectors.png")
            st.write(
                "**Stephen King and Shakespeare**? Well... It's more likely that these films were based on their literary work.")
            st.write("**Woody Allen** is an Academy Award-winning American director, writer, actor, and comedian whose career spans more than six decades and countless films.")
            st.write("**Tyler Perry** has directed some amazing films like Acrimony and the Madea films. He now has his motion picture studio, 'Tyler Perry Studios'.")
            st.write("**Luc Paul Maurice Besson** is a French film director, screenwriter, and producer. He directed or produced the films Subway, The Big Blue, and La Femme Nikita. Besson is associated with the Cinéma du look film movement.")
            st.image("resources/imgs/plot.png")
            st.write("**Comedy** and similar terms like **stand up special, stand up comedy** are quite prominent in the plot keywords of the movies. This makes it reasonable, given that **Comedy** is the second most popular movie genre.")

    # App developers page
    if page_selection == "Developers":
        st.title("Developers & Future Work")
        st.write("**STREAME** was developed by a team of talented future data scientists, data engineers, and machine learning engineers from the EXPLORE DATA SCIENCE ACADEMY. And we believe with more work in the future, **STREAME** can improve.")
        st.write("To improve the recommendations made by **STREAME** we believe we can incorporate some supervised, unsupervised algorithms such as dimensionality reduction and clustering, Matrix factorization, Single Value Decomposition, Restricted Boltzman Machines into an ensemble to predict a single output. We will look into the use of Big Data tools like Apache Hadoop for large scale data processing and use reinforcement algorithms to provide recommendations to users.")

        st.info("For any enquiries, feel free to contact any of the team members.")
        st.write("Khwatsi Hlungwani: khwatsihlungwani@gmail.com")
        st.write("Lerato Mohlala: leratomohlala93@gmail.com")
        st.write("Mukovhe Lugisani: mlugisani@gmail.com")
        st.write("Noluthando Ntsangani: noluthandontsngn@gmail.com")
        st.write("Thobani Mtshali: thobimts99@gmail.com")

        # Footer
        #st.image("resources/imgs/EDSA_logo.png", caption='Team-AM2')


# https://i1.wp.com/backgroundcheckall.com/wp-content/uploads/2017/12/netflix-background-9.jpg
# popcorn.jpeg
# netflix.mp4
# netfix.jpg
# logonetflix.jpg
# facebook_profile_image.png
# logo_transparent.png
# logo.png
# linkedin_banner_image_1.png
# twitter_header_photo_1.png
# logo_transparent_2.png

# Changing background colour

main_bg = "resources/imgs/logo_transparent_2.png"
main_bg_ext = "png"

#side_bg = "sample.jpg"
#side_bg_ext = "jpg"

st.markdown(
    f"""
    <style>
    .reportview-container {{
       background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)


if __name__ == '__main__':
    main()
