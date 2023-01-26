import flappy_bird_gym
from keras import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

"""
* AI do gry Flappy Bird *

Autorzy:
- Bartosz Krystowski s19545
- Robert Brzoskowski s21162

Przygotowanie środowiska:
Instalacja flappy_bird_gym oraz bibliotek keras, rl
"""

"""Funkcja budująca model sieci neuronowej"""
def build_model(input, actions):
    model = Sequential()
    """#Dodanie warstw sieci neuronowej"""
    model.add(Dense(64, activation='relu', input_shape=(1, input)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))

    model.add(Flatten())
    model.add(Dense(actions, activation='linear'))
    return model

"""Funkcja budująca agenta"""
def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=0.5, value_min=.0001, value_test=.0, nb_steps=6000000)
    memory = SequentialMemory(limit=100000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   enable_dueling_network=True, dueling_type='avg',
                   nb_actions=actions, nb_steps_warmup=500)
    return dqn

"""Import środowiska"""
env = flappy_bird_gym.make("FlappyBird-v0")
states = env.observation_space.shape[0]
actions = env.action_space.n

"""Stworzenie modelu"""
model = build_model(states, actions)
"""Stworzenie agenta"""
dqn = build_agent(model, actions)

"""Trenowanie sieci neuronowej"""
dqn.compile(Adam(lr=0.00025))
"""dqn.fit(env, nb_steps=5000000, visualize=True, verbose=1)"""

"""Zapis wag treningowych do pliku"""
"""dqn.save_weights('models/weights.h5')"""

"""Wczytanie wag z pliku"""
dqn.load_weights('models/7million.h5')

"""Pokazanie testowania z danymi wagami"""
dqn.test(env, visualize=True, nb_episodes=50)
env.play()
