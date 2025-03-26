from pybondmachine.converters import cnn2bm

from json.encoder import JSONEncoder
from keras import layers as klayers, models as kmodels

height = 8
width = 8
channels = 1
input_shape = (height, width, channels)

model = kmodels.Sequential()
model.add(klayers.Input(shape=input_shape))
model.add(klayers.Conv2D(2, (4, 3), activation='relu'))
model.add(klayers.MaxPool2D((2, 2)))
model.add(klayers.Flatten())
model.add(klayers.Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

encoder = JSONEncoder(indent=2)
print(encoder.encode(cnn2bm.serialize_model(model)))
