import sys
sys.path.append("game/")

from keras.models import Sequential
from keras.layers import Convolution2D, Activation, Flatten, Dense
from keras.initializations import normal
from keras.optimizers import Adam
from collections import deque
import wrapped_flappy_bird as game
import numpy as np
import skimage as skimage
from skimage import transform, color, exposure


# preprocessed image info
img_height, img_width = 80, 80
# stack 4 frames to infer the velocity information of the bird
img_frames = 4
cnn_input_shape = (img_height, img_width, img_frames)

NUM_ACTION = 2  # bird only has 2 actions: fly, or not
INITIAL_EPSILON = 0.01  # starting value of epsilon


def __init__(self, actions):
    # init replay memory
    self.replayMemory = deque()
    # init some parameters
    self.timeStep = 0
    self.epsilon = INITIAL_EPSILON
    self.actions = actions
    # init Q network
    self.create_q_net()


def create_q_net():
    """
    A five-layer convolutional network with the following architecture:

    (conv - relu) - (conv - relu) - (conv - relu) - fc - softmax

    The network operates on stack of images that have shape (H, W, N)
    consisting of N images, each with height H and width W.
    """
    model = Sequential()

    # first conv layer
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), init=lambda shape, name: normal(shape, scale=0.01, name=name), borber_mode='same', input_shape=cnn_input_shape))
    model.add(Activation('relu'))

    # second conv layer
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
    model.add(Activation('relu'))

    # third conv layer
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
    model.add(Activation('relu'))

    # full-connected layer
    model.add(Flatten())
    model.add(Dense(512, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    model.add(Activation('relu'))

    # output layer
    model.add(Dense(2, init=lambda shape, name: normal(shape, scale=0.01, name=name)))

    # compile model
    adam = Adam(lr=1e-6)
    model.compile(optimizer=adam, loss='mean_squared_error')


def train_dqn():
    game_state = game.GameState()

    # experience replay pool
    replay_pool = deque()

    # get the initial state
    init_state = get_init_state(game_state)

    t=0
    while True:
        # initialize params for Q-net
        loss = 0
        Q_sa = 0
        action_index = 0



        t+=1


def get_init_state(game_state):
    """
    1. Set the init action to "not fly"
    2. Get the initial image from wrapped_flappy_bird.py.
    3. Stack the inital image four times to form the inital state.
    :param game_state: game state to communicate with emulator
    :return: s_0 = (1, H, W, N)
    """
    a_0 = np.zeros(NUM_ACTION)
    # actions are defined in wrapped_flappy_bird.py
    # input_actions[0] == 1: do nothing
    # input_actions[1] == 1: flap the bird
    a_0[0] = 1  # do nothing
    x_0, r_0, terminal = game_state.frame_step(a_0)

    # pre-process image
    # compute luminance of an RGB image
    x_0 = skimage.color.rgb2gray(x_0)
    # resize image to match a certain size
    x_0 = skimage.transform.resize(x_0, (80, 80))
    # return image after stretching or shrinking its intensity levels
    x_0 = skimage.exposure.rescale_intensity(x_0, out_range=(0, 255))

    # stack the images to form a state, the init state consists of four same images
    s_0 = np.stack((x_0, x_0, x_0, x_0), axis=0)
    # In Keras, need to reshape
    s_0 = s_0.reshape(1, s_0.shape[0], s_0.shape[1], s_0.shape[2])
    return s_0
