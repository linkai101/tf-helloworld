# tf-helloworld
A basic chatbot I made using Tensorflow.

## Training model
Training data is stored in `intents.json`. To train the model, run the `training.ipynb` Jupyter Notebook file. This will dump a `model.h5` file.

## Running the bot
Run the `bot.py` or `bot.ipynb` file. This takes in the trianed model from `model.h5` and runs a basic chatbot UI.