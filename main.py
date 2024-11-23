from tkinter import UNITS
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import customtkinter as ctk
import os

#daca exista o retea salvata deja intreb utilizatorul daca nu vrea acea retea
neural_network_path = 'retea_numere.h5'

while True:
    try:
        load = input('Doriti sa se caute o reta deja invatata si sa o folositi? y/n :')
        if load.lower() not in ('y', 'n'):
            raise ValueError ("Pune doar litera 'y' pentru da sau 'n' pentru nu.")
        break
    except ValueError as e:
        print(e)

if (os.path.exists(neural_network_path)) and load == 'y':
        neural_network = tf.keras.models.load_model(neural_network_path)
        print('reteaua a fost incarcata cu succes\n')    
else: 
    print("Nu ati ales sa se caute retea sau nu exista nici o retea")
    print('Se va antrena o retea...\n')

    #importez dataset-ul cu numerele scrise de mana
    mnist = tf.keras.datasets.mnist

    #impart datele in date de antrenament si date de testare
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    #nomralizez valorile 
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    #modelul retelei este urmatorul:
    #un strat de intrare
    #doua straturi ascunse 
    #un strat de iesire

    neural_network = tf.keras.models.Sequential()

    neural_network.add(tf.keras.layers.Flatten(input_shape=(28,28)))

    neural_network.add(tf.keras.layers.Dense(units=36, activation=tf.nn.relu))
    neural_network.add(tf.keras.layers.Dense(units=36, activation=tf.nn.relu))

    neural_network.add(tf.keras.layers.Dense(units=10, activation=tf.keras.activations.softmax))

    #compilez reteua neuronala inainte de invatare
    neural_network.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    #pun reteaua sa invete
    neural_network.fit(x_train, y_train, epochs=10)

    print('acum incepe procesul de testare')

    #evaluez reteaua neuronala
    loss, accuracy = neural_network.evaluate(x_test, y_test)
    print(f'acuratetea este {accuracy} \n eroare este de {loss}')

    #salvez tariile sinaptice
    while True:
        try:
            save = input('Doriti sa salvati tariile sinaptice ale acestei retele? y/n :')
            if save.lower() not in ('y', 'n'):
                raise ValueError("Pune doar litera 'y' pentru da sau 'n' pentru nu.")
            break
        except ValueError as e:
            print(e)
    if save == 'y' :
        #try:
        neural_network.save('retea_numere.h5')
            #print("reteuaua a fost salvata cu succes")
        #except:
                #print("salvarea nu a putut fi efecutata")
    else: 
        print("Nu ati dorit salvarea retelei")


#citesc imaginile create de mine
for x in range(1,7):
    img = cv.imread(f'{x}.png')[:,:,0]
    img = np.invert(np.array([img]))
    #cer rezultatul de la reteaua neuronala
    prediction = neural_network.predict(img)
    print(f'numarul din imagine este :{np.argmax(prediction)}\n')
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()