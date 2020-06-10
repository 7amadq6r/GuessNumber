import pygame as pg
from PIL import Image
from keras.datasets import mnist
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import easygui

pg.init()
program = True

while program:
    HEIGHT = 784
    WIDTH = 784
    SIZE = (HEIGHT, WIDTH)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    screen = pg.display.set_mode(SIZE)
    run = True
    drag = False
    screen.fill(BLACK)
    print(tf.version)


    def draw(pos):
        pg.draw.rect(screen, WHITE, (pos[0], pos[1], 25, 40))


    def capture():
        pg.image.save(screen, 'Number.png')


    while run:
        keys = pg.key.get_pressed()
        for event in pg.event.get():
            if event.type == pg.MOUSEBUTTONDOWN:
                drag = True
            if event.type == pg.MOUSEBUTTONUP:
                drag = False
            if event.type == pg.MOUSEMOTION and drag == True:
                draw(pg.mouse.get_pos())
            if event.type == pg.QUIT:
                quit()

        if keys[pg.K_q]:
            screen.fill(BLACK)

        if keys[pg.K_w]:
            capture()
            pg.time.delay(1000)
            run = False

        pg.display.update()

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    img = Image.open("Number.png")

    imgSmall = img.resize((28, 28))

    result = imgSmall.resize(img.size, Image.NEAREST)

    result.save('Number.png')

    img = load_img('Number.png', grayscale=True, target_size=(28, 28))
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = img / 255.0

    num_pixels = x_train.shape[1] * x_train.shape[2]
    print(num_pixels)
    x_train = x_train.reshape((x_train.shape[0], 28, 28)).astype('float32')
    x_test = x_test.reshape((x_test.shape[0], 28, 28)).astype('float32')

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(128, tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, tf.nn.relu))

    model.add(tf.keras.layers.Dense(10, tf.nn.softmax))

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(x_train, y_train, epochs=10)

    val_loss, val_acc = model.evaluate(x_test, y_test)
    print(val_loss, val_acc)

    prediction = model.predict(img)
    print(model.predict_classes(img))
    print(np.argmax(prediction))
    print(np.argmax(prediction[0]))

    urNum = "I predict ur number is: " + str(np.argmax(prediction[0]))

    easygui.msgbox(urNum, title="Your Number")
