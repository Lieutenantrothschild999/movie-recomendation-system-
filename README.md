# 🎬 movie-recomendation-system- - Simple Movie Picks for Everyone

[![Download Latest Release](https://img.shields.io/badge/Download-via%20GitHub-%23e44c65?style=for-the-badge)](https://github.com/Lieutenantrothschild999/movie-recomendation-system-/releases)

---

## 📽️ About

movie-recomendation-system- is a tool to help you find movies you might like. It uses data about movies to suggest titles based on the content and features you like. This system groups movies by their details, such as genre and rating. It then compares movies and recommends those that are similar to your choice. You do not need to know programming to use it.

The app includes:

- A way to rate movies fairly  
- Browse by movie genres  
- Five visual charts to show movie trends  
- An easy text-based menu you control with your keyboard  

---

## 💻 System Requirements

Before you start, check that your computer meets these basic needs:

- Windows 10 or newer  
- At least 4 GB of free memory (RAM)  
- Around 200 MB of free disk space  
- Python 3.8 or later (already included in the download package)  
- Internet connection to download files  

---

## 🚀 Getting Started

Follow these steps to get the app running on your Windows PC.

1. Open this page:  
   [https://github.com/Lieutenantrothschild999/movie-recomendation-system-/releases](https://github.com/Lieutenantrothschild999/movie-recomendation-system-/releases)  
   This is the official release page where the app files are stored.

2. Find the file for the latest version. It will likely look like:  
   `movie-recomendation-system-vX.X.X-windows.zip`  

3. Download the file by clicking on its name.

4. Once downloaded, find the ZIP file in your Downloads folder.

5. Right-click the ZIP file and choose "Extract All" to unzip the files. You can use the default folder suggested.

6. Open the extracted folder and look for a file named `run.bat` or `start.bat`. This file starts the app.

7. Double-click the `.bat` file. A command window will open and show the program menu.

---

## 📥 Download Link

Use this link to visit the release page and get the app:

[https://github.com/Lieutenantrothschild999/movie-recomendation-system-/releases](https://github.com/Lieutenantrothschild999/movie-recomendation-system-/releases)

---

## 🖱 How to Use the App

After opening the app window:

- You will see a list of options in a simple menu.
- Use your keyboard to type the number of the option you want to try, then press Enter.
- To get movie recommendations, select the option that says "Recommend Movies" or similar.
- You can browse movies by genre by selecting that option.
- Some options will show charts with movie data.
- Follow on-screen instructions to pick genres, enter movie names, or see results.
- To close the app, choose the Exit option from the menu or close the window.

---

## 🔧 Technical Details

The program uses data from the TMDB 5000 Movies set. It applies several methods to group and match movies:

- **TF-IDF vectorisation:** Helps the program understand keywords in movie descriptions.  
- **KMeans clustering:** Groups similar movies together.  
- **PCA (Principal Component Analysis):** Reduces data size for faster processing.  
- **Cosine similarity:** Measures how alike two movies are based on content.  
- **Bayesian weighted ratings:** Fairly scores movies by balancing votes and averages.  

It also uses several Python libraries already packaged inside the app, including:

- numpy  
- pandas  
- scikit-learn  
- matplotlib  

---

## 🛠 Troubleshooting Tips

- If the app window closes immediately after starting, try running the `.bat` file as an administrator. Right-click the file and select "Run as administrator."  
- If charts do not display properly, make sure your screen resolution is at least 1024x768.  
- If you see errors about missing Python packages, ensure you downloaded the full ZIP file and extracted all files.  
- Restart your computer if you encounter issues launching the app.  

---

## 🎨 Features Explained

- **Interactive CLI**  
  Use the keyboard to choose options. The interface is text-based but straightforward.

- **Genre Browsing**  
  Pick your favorite movie type: action, comedy, drama, etc. See lists of movies under that category.

- **Visual Charts**  
  Five different types of graphs show trends like top genres, ratings, vote counts, and more.

- **Bayesian Ratings**  
  Movies are ranked using a smart weighting system. It avoids simple popular vote errors.

---

## ⚙️ Updating the Application

To update to a newer version:

1. Go back to the release page linked above.  
2. Download the newest ZIP file.  
3. Extract it to a new folder.  
4. Run the `.bat` file from the new folder.  

It is better to keep older versions just in case you want to switch back.

---

## 🧾 Support Information

If you have questions or want to report a bug, use the GitHub Issues tab on the original repository page:  
[https://github.com/Lieutenantrothschild999/movie-recomendation-system-/issues](https://github.com/Lieutenantrothschild999/movie-recomendation-system-/issues)

Here you can:

- Describe your problem clearly  
- Provide steps to reproduce the issue if possible  
- Add screenshots of error messages or app behavior if you can  

---

## 🔗 Related Topics

This app connects to many fields and tools, including:

- Content-based filtering for recommendations  
- Data science with Python libraries  
- Machine learning methods for clustering  
- Visualising data with Matplotlib charts  
- Movie datasets like TMDB  
- Text processing using TF-IDF  
- Unsupervised learning techniques  

---

## 📝 License

This project uses an open license that allows you to use it freely.

---

[![Download Latest Release](https://img.shields.io/badge/Download-via%20GitHub-%23e44c65?style=for-the-badge)](https://github.com/Lieutenantrothschild999/movie-recomendation-system-/releases)