import csv

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


X = []
Y = []

cesta = r"C:\Users\st025537\Downloads\kb4b_prog\3. strojove_uceni\data\Most Streamed Spotify Songs 2024.csv"
with open(cesta, "r", encoding="ISO-8859-1") as file:
        for radek in csv.DictReader(file):
            try:
                spotify_streams = float(radek["Spotify Streams"].replace(",", "")) 
                youtube_views = float(radek["YouTube Views"].replace(",", "")) 
                tiktok_views = float(radek["TikTok Views"].replace(",", "")) 
                pandora_streams = float(radek["Pandora Streams"].replace(",", "")) 
                tiktok_posts = float(radek["TikTok Posts"].replace(",", ""))
                tiktok_views = float(radek["TikTok Views"].replace(",", ""))

            except ValueError:
                continue
            
            streams = (
            spotify_streams +
            youtube_views +
            pandora_streams
            )

            if tiktok_views > 100_000_000:
                Y.append(1)  # Track has more TikTok views than 100 000 000
            else:
                Y.append(0)  # Track has less streams than 100 000 000

            X.append([spotify_streams,youtube_views,pandora_streams])

X_train = X[:round(0.8*len(X))]
Y_train = Y[:round(0.8*len(Y))]

X_test = X[round(0.8*len(X)):]
Y_test = Y[round(0.8*len(Y)):]


neuronka = MLPClassifier(
    hidden_layer_sizes=(10,4),
    activation="relu",
    max_iter=5_000,
    n_iter_no_change=10
)
neuronka.fit(X_train, Y_train)


predikce = neuronka.predict(X_test)
pocet = len(predikce)


percentage = (sum(Y_train) / len(Y_train)) * 100
print(f"Percentage of tracks with more TikTok views than 100 000 000: {percentage:.2f}%")


correct = 0
for i in range(len(predikce)):
    if Y_train[i] == predikce[i]:
        correct += 1
print("Accuracy:", correct / len(predikce))

ConfusionMatrixDisplay.from_predictions(
    Y_test, predikce)
plt.show()