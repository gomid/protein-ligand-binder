from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv3D, MaxPool3D, concatenate

def build_model(input_shape=(400, 400, 400, 2)):
    input_protein = Input(shape=input_shape)
    x = Conv3D(filters=16, kernel_size=(64, 64, 64), padding='Same', activation='relu')(input_protein)
    x = Conv3D(filters=32, kernel_size=(16, 16, 16), padding='Same', activation='relu')(x)
    x = Conv3D(filters=64, kernel_size=(4, 4, 4), padding='Same', activation='relu')(x)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)

    input_ligand = Input(shape=input_shape)
    y = Conv3D(filters=16, kernel_size=(64, 64, 64), padding='Same', activation='relu')(input_ligand)
    y = Conv3D(filters=32, kernel_size=(16, 16, 16), padding='Same', activation='relu')(y)
    y = Conv3D(filters=64, kernel_size=(4, 4, 4), padding='Same', activation='relu')(y)
    y = MaxPool3D(pool_size=(2, 2, 2))(y)
    y = Dropout(0.25)(y)
    y = Flatten()(y)

    x = concatenate([x, y])
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    out = Dense(1, activation="sigmoid")(x)
    return Model(inputs=[input_protein, input_ligand], outputs=out)
