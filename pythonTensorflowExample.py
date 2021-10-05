import numpy as np
import tensorflow as tf
from numpy import genfromtxt

goodAPoses = genfromtxt('/Users/luca/Desktop/workingDirectory/previewImages/GoodAPose/poses.csv', delimiter=',')
badAPoses = genfromtxt('/Users/luca/Desktop/workingDirectory/previewImages/BadAPose/poses.csv', delimiter=',')

labelsGood = np.ones([goodAPoses.shape[0]], dtype=np.int32)
labelsBad = np.zeros([badAPoses.shape[0]], dtype=np.int32)

X = np.concatenate((goodAPoses, badAPoses), axis=0)
Y = np.concatenate((labelsGood, labelsBad), axis=0)

X = tf.keras.utils.normalize(X, axis=-1, order=2)

model = tf.keras.models.Sequential([
    # tf.keras.Input(shape=(34,)),
  tf.keras.layers.Dense(34, input_shape=(34,), activation='relu', kernel_initializer='he_uniform'),
  tf.keras.layers.Dense(17, activation='relu', kernel_initializer='he_uniform'),
  tf.keras.layers.Dense(17, activation='relu', kernel_initializer='he_uniform'),
  tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer='he_uniform')
])

# loss_fn1 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss=loss_fn,
            metrics=['accuracy'])
              # metrics=tf.keras.metrics.BinaryCrossentropy())

# inputs = tf.keras.Input(shape=(34,))
# x = tf.keras.layers.Dense(34, activation=tf.nn.relu)(inputs)
# x = tf.keras.layers.Dense(34, activation=tf.nn.relu)(x)
# x = tf.keras.layers.Dense(17, activation=tf.nn.relu)(x)
# outputs = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(x)
# model = tf.keras.Model(inputs=inputs, outputs=outputs)

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=["accuracy"])

model.summary()

model.fit(X, Y, epochs=10, validation_split=0.2, batch_size=100, verbose=0)

goodAPose = np.array([0.101318,0.491829,0.0852958,0.511805,0.0841727,0.471087,0.106221,0.534901,0.102035,0.447689,0.214236,0.577794,0.205027,0.401328,0.307764,0.69233,0.296297,0.276292,0.38414,0.789242,0.369121,0.182343,0.478153,0.548575,0.479801,0.430937,0.712054,0.564377,0.709802,0.417897,0.900895,0.570023,0.907556,0.407681])
badAPose = np.array([0.0855109,0.488726,0.0764183,0.507854,0.075791,0.470685,0.106666,0.533184,0.104461,0.45109,0.210827,0.58181,0.220695,0.417511,0.347723,0.630027,0.350974,0.381943,0.457796,0.6642,0.451945,0.327834,0.475397,0.556228,0.474085,0.431436,0.713491,0.579555,0.707039,0.404139,0.911279,0.60715,0.896295,0.384118])
test = X[0]
test2 = X[-1]
a1 = goodAPose[np.newaxis, :]
a2 = badAPose[np.newaxis, :]
print(model.predict(a1))
print(model.predict(a2))



# example_id = numpy.array(['%d' % i for i in range(len(Y))])
#
# x_column_name = 'x'
# example_id_column_name = 'example_id'
# tf.compat.v2.
# train_input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={x_column_name: X, example_id_column_name: example_id},
#     y=Y,
#     num_epochs=None,
#     shuffle=True)
#
# svm = tf.contrib.learn.SVM(
#     example_id_column=example_id_column_name,
#     feature_columns=(tf.contrib.layers.real_valued_column(
#         column_name=x_column_name, dimension=34),),
#     l2_regularization=0.1)
#
# svm.fit(input_fn=train_input_fn, steps=10)