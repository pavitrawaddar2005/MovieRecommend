import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
from PIL import Image, ImageTk
import requests
from io import BytesIO

class MovieRecommendationSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Movie Recommendation System")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2c3e50')
        
        # Create sample movie data
        self.movies = self.generate_sample_data()
        
        # Initialize the TF-IDF Vectorizer
        self.tfidf = TfidfVectorizer(stop_words='english')
        
        # Construct the TF-IDF matrix
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies['description'])
        
        # Compute the cosine similarity matrix
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        
        # Create GUI
        self.create_gui()
        
    def generate_sample_data(self):
        """Generate sample movie data for demonstration"""
        movies = pd.DataFrame({
            'title': [
                'The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 
                'Pulp Fiction', 'Forrest Gump', 'Inception', 'The Matrix',
                'Goodfellas', 'The Silence of the Lambs', 'Star Wars: A New Hope',
                'The Lord of the Rings: The Fellowship of the Ring', 'Fight Club',
                'Parasite', 'Interstellar', 'The Avengers', 'Toy Story',
                'Spirited Away', 'The Lion King', 'Alien', 'Gladiator'
            ],
            'genre': [
                'Drama', 'Crime, Drama', 'Action, Crime, Drama', 'Crime, Drama',
                'Drama, Romance', 'Action, Adventure, Sci-Fi', 'Action, Sci-Fi',
                'Biography, Crime, Drama', 'Crime, Drama, Thriller', 'Action, Adventure, Fantasy',
                'Adventure, Drama, Fantasy', 'Drama', 'Comedy, Drama, Thriller',
                'Adventure, Drama, Sci-Fi', 'Action, Adventure, Sci-Fi', 'Animation, Adventure, Comedy',
                'Animation, Adventure, Family', 'Animation, Adventure, Drama', 'Horror, Sci-Fi',
                'Action, Adventure, Drama'
            ],
            'year': [
                1994, 1972, 2008, 1994, 1994, 2010, 1999, 1990, 1991, 1977,
                2001, 1999, 2019, 2014, 2012, 1995, 2001, 1994, 1979, 2000
            ],
            'rating': [9.3, 9.2, 9.0, 8.9, 8.8, 8.8, 8.7, 8.7, 8.6, 8.6, 
                      8.8, 8.8, 8.6, 8.6, 8.0, 8.3, 8.6, 8.5, 8.4, 8.5],
            'description': [
                'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
                'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.',
                'When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.',
                'The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.',
                'The presidencies of Kennedy and Johnson, the events of Vietnam, Watergate, and other historical events unfold through the perspective of an Alabama man with an IQ of 75.',
                'A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.',
                'A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.',
                'The story of Henry Hill and his life in the mob, covering his relationship with his wife Karen Hill and his mob partners Jimmy Conway and Tommy DeVito.',
                'A young F.B.I. cadet must receive the help of an incarcerated and manipulative cannibal killer to help catch another serial killer.',
                'Luke Skywalker joins forces with a Jedi Knight, a cocky pilot, a Wookiee and two droids to save the galaxy from the Empire\'s world-destroying battle station.',
                'A meek Hobbit from the Shire and eight companions set out on a journey to destroy the powerful One Ring and save Middle-earth from the Dark Lord Sauron.',
                'An insomniac office worker and a devil-may-care soapmaker form an underground fight club that evolves into something much, much more.',
                'Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan.',
                'A team of explorers travel through a wormhole in space in an attempt to ensure humanity\'s survival.',
                'Earth\'s mightiest heroes must come together and learn to fight as a team if they are going to stop the mischievous Loki and his alien army from enslaving humanity.',
                'A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boy\'s room.',
                'During her family\'s move to the suburbs, a sullen 10-year-old girl wanders into a world ruled by gods, witches, and spirits.',
                'Lion prince Simba and his father are targeted by his bitter uncle, who wants to ascend the throne himself.',
                'The crew of a commercial spacecraft encounter a deadly lifeform after investigating an unknown transmission.',
                'A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family and sent him into slavery.'
            ]
        })
        return movies
    
    def create_gui(self):
        """Create the graphical user interface"""
        # Title
        title_label = tk.Label(self.root, text="Movie Recommendation System", 
                              font=("Arial", 20, "bold"), bg='#2c3e50', fg='white')
        title_label.pack(pady=20)
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left frame for inputs
        left_frame = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Right frame for recommendations
        right_frame = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Input section
        input_label = tk.Label(left_frame, text="Select a Movie You Like:", 
                              font=("Arial", 12, "bold"), bg='#34495e', fg='white')
        input_label.pack(pady=10)
        
        # Movie selection dropdown
        self.selected_movie = tk.StringVar()
        movie_dropdown = ttk.Combobox(left_frame, textvariable=self.selected_movie, 
                                     values=list(self.movies['title']), state="readonly")
        movie_dropdown.pack(pady=5, padx=10, fill=tk.X)
        movie_dropdown.set("Select a movie")
        
        # Genre preference
        genre_label = tk.Label(left_frame, text="Preferred Genre (optional):", 
                              font=("Arial", 12, "bold"), bg='#34495e', fg='white')
        genre_label.pack(pady=(20, 5))
        
        # Get unique genres
        all_genres = set()
        for genres in self.movies['genre']:
            for genre in genres.split(', '):
                all_genres.add(genre)
        
        self.selected_genre = tk.StringVar()
        genre_dropdown = ttk.Combobox(left_frame, textvariable=self.selected_genre, 
                                     values=list(all_genres), state="readonly")
        genre_dropdown.pack(pady=5, padx=10, fill=tk.X)
        genre_dropdown.set("Any genre")
        
        # Rating filter
        rating_label = tk.Label(left_frame, text="Minimum Rating:", 
                               font=("Arial", 12, "bold"), bg='#34495e', fg='white')
        rating_label.pack(pady=(20, 5))
        
        self.rating_var = tk.DoubleVar(value=7.0)
        rating_scale = tk.Scale(left_frame, variable=self.rating_var, from_=0, to=10, 
                               resolution=0.5, orient=tk.HORIZONTAL, bg='#34495e', fg='white')
        rating_scale.pack(pady=5, padx=10, fill=tk.X)
        
        # Number of recommendations
        count_label = tk.Label(left_frame, text="Number of Recommendations:", 
                              font=("Arial", 12, "bold"), bg='#34495e', fg='white')
        count_label.pack(pady=(20, 5))
        
        self.count_var = tk.IntVar(value=5)
        count_spinbox = tk.Spinbox(left_frame, from_=1, to=10, textvariable=self.count_var, 
                                  width=10, bg='#ecf0f1')
        count_spinbox.pack(pady=5, padx=10)
        
        # Recommendation button
        recommend_btn = tk.Button(left_frame, text="Get Recommendations", 
                                 command=self.get_recommendations, bg='#3498db', fg='white',
                                 font=("Arial", 12, "bold"), relief=tk.RAISED, bd=2)
        recommend_btn.pack(pady=20, padx=10, fill=tk.X)
        
        # Results section
        results_label = tk.Label(right_frame, text="Recommended Movies", 
                                font=("Arial", 16, "bold"), bg='#34495e', fg='white')
        results_label.pack(pady=10)
        
        # Treeview for recommendations
        columns = ('title', 'genre', 'year', 'rating')
        self.tree = ttk.Treeview(right_frame, columns=columns, show='headings', height=15)
        
        # Define headings
        self.tree.heading('title', text='Title')
        self.tree.heading('genre', text='Genre')
        self.tree.heading('year', text='Year')
        self.tree.heading('rating', text='Rating')
        
        # Define column widths
        self.tree.column('title', width=250)
        self.tree.column('genre', width=150)
        self.tree.column('year', width=80)
        self.tree.column('rating', width=80)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        
        # Pack tree and scrollbar
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
        
        # Bind selection event
        self.tree.bind('<<TreeviewSelect>>', self.on_movie_select)
        
        # Movie details frame
        self.details_frame = tk.Frame(right_frame, bg='#34495e')
        self.details_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.details_text = tk.Text(self.details_frame, height=6, width=60, bg='#2c3e50', 
                                   fg='white', font=("Arial", 10), wrap=tk.WORD)
        self.details_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.details_text.config(state=tk.DISABLED)
    
    def get_recommendations(self):
        """Get movie recommendations based on selected movie and filters"""
        selected_movie = self.selected_movie.get()
        
        if selected_movie == "Select a movie":
            messagebox.showwarning("Input Error", "Please select a movie first!")
            return
        
        # Clear previous recommendations
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Get recommendations
        recommendations = self.recommend_movies(selected_movie)
        
        # Apply genre filter if specified
        preferred_genre = self.selected_genre.get()
        if preferred_genre != "Any genre":
            recommendations = recommendations[
                recommendations['genre'].str.contains(preferred_genre)
            ]
        
        # Apply rating filter
        min_rating = self.rating_var.get()
        recommendations = recommendations[recommendations['rating'] >= min_rating]
        
        # Limit to requested number
        num_recommendations = self.count_var.get()
        recommendations = recommendations.head(num_recommendations)
        
        # Add to treeview
        for _, row in recommendations.iterrows():
            self.tree.insert('', tk.END, values=(
                row['title'], row['genre'], row['year'], row['rating']
            ))
    
    def recommend_movies(self, title, cosine_sim=None):
        """Get movie recommendations based on content similarity"""
        if cosine_sim is None:
            cosine_sim = self.cosine_sim
        
        # Get the index of the movie that matches the title
        idx = self.movies[self.movies['title'] == title].index[0]
        
        # Get the pairwise similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[1:11]
        
        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Return the top 10 most similar movies
        return self.movies.iloc[movie_indices]
    
    def on_movie_select(self, event):
        """Show details of selected movie"""
        selected_item = self.tree.selection()
        if not selected_item:
            return
        
        # Get movie title from selected row
        item = self.tree.item(selected_item[0])
        title = item['values'][0]
        
        # Get movie details
        movie = self.movies[self.movies['title'] == title].iloc[0]
        
        # Display details
        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(tk.END, f"Title: {movie['title']}\n")
        self.details_text.insert(tk.END, f"Year: {movie['year']}\n")
        self.details_text.insert(tk.END, f"Genre: {movie['genre']}\n")
        self.details_text.insert(tk.END, f"Rating: {movie['rating']}/10\n\n")
        self.details_text.insert(tk.END, f"Description:\n{movie['description']}")
        self.details_text.config(state=tk.DISABLED)

# Create the main window
if __name__ == "__main__":
    root = tk.Tk()
    app = MovieRecommendationSystem(root)
    root.mainloop()