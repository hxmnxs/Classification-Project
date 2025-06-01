from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from sklearn.utils import class_weight

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, early_stopping_patience, use_class_weights=True):
    callbacks = []
    if early_stopping_patience > 0:
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=early_stopping_patience, restore_best_weights=True)
        callbacks.append(early_stop)

    class_weights_dict = None
    if use_class_weights:
        try:
            weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weights_dict = dict(enumerate(weights))
            print(f"Using class weights: {class_weights_dict}")
        except Exception as e:
            print(f"Could not compute class weights: {e}. Proceeding without class weights.")
            class_weights_dict = None


    validation_data = None
    if X_val is not None and not X_val.empty and y_val is not None and not y_val.empty:
        validation_data = (X_val, y_val)
    else:
        print("No validation data provided or it's empty. Training without validation callback monitoring based on val_loss (if EarlyStopping is used).")
        # Modify EarlyStopping if no validation data and it monitors val_loss
        if early_stopping_patience > 0 and callbacks and isinstance(callbacks[0], EarlyStopping):
            if callbacks[0].monitor == 'val_loss':
                print("EarlyStopping monitor is 'val_loss' but no validation data. Consider changing monitor to 'loss' or removing EarlyStopping.")
                # For simplicity, we'll let it run; Keras might warn or error if val_loss is monitored without validation_data.
                # Or, remove the callback: callbacks = []

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=callbacks,
        class_weight=class_weights_dict,
        verbose=1
    )
    
    return model, history

if __name__ == '__main__':
    # This is a mock example.
    # You would need processed data and a built model to run this directly.
    from model_builder import build_ann_model
    from config import LEARNING_RATE, ANN_LAYER_CONFIG, OUTPUT_LAYER_ACTIVATION, LOSS_FUNCTION, METRICS, EPOCHS, BATCH_SIZE, EARLY_STOPPING_PATIENCE

    # Dummy data for testing trainer
    X_train_sample = pd.DataFrame(np.random.rand(100, 10))
    y_train_sample = pd.Series(np.random.randint(0, 2, 100))
    X_val_sample = pd.DataFrame(np.random.rand(20, 10))
    y_val_sample = pd.Series(np.random.randint(0, 2, 20))
    
    sample_model = build_ann_model(
        input_shape=X_train_sample.shape[1],
        learning_rate=LEARNING_RATE,
        layer_config=ANN_LAYER_CONFIG,
        output_activation=OUTPUT_LAYER_ACTIVATION,
        loss_function=LOSS_FUNCTION,
        metrics_list=METRICS
    )
    
    print("Training dummy model...")
    trained_model, history_obj = train_model(
        sample_model, X_train_sample, y_train_sample, X_val_sample, y_val_sample,
        epochs=3, # Few epochs for quick test
        batch_size=BATCH_SIZE,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        use_class_weights=True
    )
    
    print("Dummy model training complete.")
    if history_obj:
        print("Training history:")
        print(history_obj.history)