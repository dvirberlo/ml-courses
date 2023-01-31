import tensorflow as tf


def main():
    print('hi')
    layer_1 = tf.keras.layers.Dense(
        units=1,
        activation=tf.keras.activations.sigmoid,
    )
    print(layer_1.get_weights())
    print(layer_1(tf.constant([[1.0, 2.0, 3.0]])))
    print(layer_1.get_weights())


if __name__ == '__main__':
    main()
