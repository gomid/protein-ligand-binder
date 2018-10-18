from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv3D, MaxPool3D, concatenate


# grid -> CNN based model -> [x, y, z] of the ligand
def build_model(input_shape=(21, 21, 21, 2)):
    input_grid = Input(shape=input_shape)
    x = Conv3D(filters=16, kernel_size=(8, 8, 8), padding='valid', activation='relu')(input_grid)
    x = Conv3D(filters=32, kernel_size=(4, 4, 4), padding='valid', activation='relu')(x)
    x = Conv3D(filters=64, kernel_size=(2, 2, 2), padding='valid', activation='relu')(x)
    x = MaxPool3D(pool_size=(2, 2, 2), padding='valid')(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)

    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    out = Dense(4, activation="sigmoid")(x)
    return Model(inputs=input_grid, outputs=out)
