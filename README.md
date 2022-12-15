Implementation of [Deep Chess](https://arxiv.org/pdf/1711.09667.pdf) using [this dataset](http://ccrl.chessdom.com/ccrl/4040/games.html)

In the end I wasn't able to get a model that actually played chess well.

Pipeline:
1. Pickle all chess game objects from pgn file because reading a large quantity takes forever (pickleData.py)
2. Convert the pickles into a h5 table (convertPicklesToData.py)
3. Train the models (trainModels.py)
4. Play against the network (playGame.py)

**Can play against my deepChess by running play game as is**

Related Videos:
[Presentation](https://youtu.be/Z2KsW2RBqhU)
[Training Demo](https://youtu.be/jv5CyTSRb5k)
[Playing Demo](https://youtu.be/dtFgmZ37zkA)