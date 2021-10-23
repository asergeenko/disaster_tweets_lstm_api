import os
from tensorflow.keras.models import Sequential, load_model

from core import get_model, config


def test_train_model():
    model = get_model()
    assert isinstance(model, Sequential)


def test_serialize_model():
    config['SAVED_MODEL_PATH'] = 'tmp_model'
    model = get_model()
    assert os.path.exists(config['SAVED_MODEL_PATH'])
    model1 = load_model(config['SAVED_MODEL_PATH'])
    assert isinstance(model1, Sequential)

