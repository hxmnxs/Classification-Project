from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from config import RANDOM_SEED

tf.random.set_seed(RANDOM_SEED) # For Keras reproducibility

def build_ann_model(input_shape, learning_rate, layer_config, output_activation, loss_function, metrics_list):
    model = Sequential()
    
    # Input layer
    model.add(Dense(layer_config[0]['units'], activation=layer_config[0]['activation'], input_shape=(input_shape,)))
    if layer_config[0].get('dropout_rate', 0) > 0:
        model.add(Dropout(layer_config[0]['dropout_rate']))
        
    # Hidden layers
    for layer_conf in layer_config[1:]:
        model.add(Dense(layer_conf['units'], activation=layer_conf['activation']))
        if layer_conf.get('dropout_rate', 0) > 0:
            model.add(Dropout(layer_conf['dropout_rate']))
            
    # Output layer
    model.add(Dense(1, activation=output_activation)) # 1 unit for binary classification
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics_list)
    
    return model

if __name__ == '__main__':
    from config import LEARNING_RATE, ANN_LAYER_CONFIG, OUTPUT_LAYER_ACTIVATION, LOSS_FUNCTION, METRICS
    
    sample_input_shape = 10 # Example input shape
    
    test_model = build_ann_model(
        input_shape=sample_input_shape,
        learning_rate=LEARNING_RATE,
        layer_config=ANN_LAYER_CONFIG,
        output_activation=OUTPUT_LAYER_ACTIVATION,
        loss_function=LOSS_FUNCTION,
        metrics_list=METRICS
    )
    
    print("Model built successfully:")
    test_model.summary()