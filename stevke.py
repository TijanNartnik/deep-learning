# importi za keras in MNIST - vse že znotraj TenserFlowa
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
# numpy je obvezen ker keras uporablja numpy za arraye
import numpy as np

# importi za slike
from PIL import Image
from PIL import ImageOps

# import teste
import os


# Kaj je MNIST Dataset?
# -> Največja baza slik in podatkov ročno napisanih številk, ki se uporablja
#    za treniranje in testiranje različnih modelov strojnega in globokega učenja.
#
# Kaj je TenserFlow?
# -> odprtokodna knjižnjica, s fokusom na globoke nevronske mreže in strojno učenje


# klasična sintaksa za python classe
class mnist_network():
    def __init__(self):
        """ naloži podatke, ustvari in strenira model """
        # naloži podatke
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        # splošči 28*28 slike v vektor 784
        num_pixels = X_train.shape[1] * X_train.shape[2]
        X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
        X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')
        # normalizira vnose z 0-255 na 0-1
        X_train = X_train / 255
        X_test = X_test / 255
        # en izhod vročega kodiranja
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        num_classes = y_test.shape[1]


        # ustvari model
        self.model = Sequential()
        self.model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
        self.model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
        # compila model
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # natrenira model
        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

        self.train_img = X_train
        self.train_res = y_train
        self.test_img = X_test
        self.test_res = y_test


    def predict_result(self, img, num_pixels = None, show=False):
        """ napove število na sliki (vektor) """
        assert type(img) == np.ndarray and img.shape == (784,)

        num_pixels = img.shape[0]
        # realna "ta prava" stevka
        res_number = np.argmax(self.model.predict(img.reshape(-1,num_pixels)), axis = 1)
        # verjetnosti
        res_probabilities = self.model.predict(img.reshape(-1,num_pixels))

        # potrebujemo le prvi element ker je samo en (stevka)
        return (res_number[0], res_probabilities.tolist()[0])    

    def partial_img_rec(self, image, upper_left, lower_right, results=[]):
        """ partial je del slike """
        left_x, left_y = upper_left
        right_x, right_y = lower_right

        print("trenutni testni del: ", upper_left, lower_right)
        print("rezultat: ", results)
        # pogoj za zaustavitev rekurzije: dosegli smo celotno širino slike
        width, height = image.size
        if right_x > width:
            return results

        partial = image.crop((left_x, left_y, right_x, right_y))
        # spremeni velikost slike na dimenzijo 28 * 28
        partial = partial.resize((28,28), Image.ANTIALIAS)

        partial.show()
        # pretvori v vector
        partial =  ImageOps.invert(partial)
        partial = np.asarray(partial, "float32")
        partial = partial / 255.
        partial[partial < 0.5] = 0.
        # splošči sliko na 28*28 = 784 vektor
        num_pixels = partial.shape[0] * partial.shape[1]
        partial = partial.reshape(num_pixels)

        step = height // 10
        #ali je na tem delu slike številka?

        res, prop = self.predict_result(partial)
        print("rezultat: ", res, ". verjetnosti: ", prop)
        # štejte ta rezultat le, če je omrežje >= 50 % prepričano
        if prop[res] >= 0.5:        
            results.append(res)
            # korak je 80 % velikosti delne slike (kar je enakovredno višini izvirne slike)
            step = int(height * 0.8)
            print("našel veljaven rezultat")
        else:
            # če številke ne najdemo, naredimo manjše korake
            step = height // 20 
        print("korak: ", step)
        # rekurzivni klic s spremenjenimi položaji (premik po spremenljivkah korakov)
        return self.partial_img_rec(image, (left_x+step, left_y), (right_x+step, right_y), results=results)

    def test_individual_digits(self):
        """ preizkusi partial_img_rec z posameznimi števkami (kvadratne slike)
            shranjenimi v mapo 'individual_tests' po vzorcu 'stevka_na_sliki'.jpg (1.jpg) """
        cnt_right, cnt_wrong = 0,0
        folder_content = os.listdir(".\individual_tests")

        for imageName in folder_content:
            print("\n--- NASLEDNJA SLIKA", imageName, " ---")
            # datoteka slike mora biti jpg ali png
            assert imageName[-4:] == ".jpg" or imageName[-4:] == ".png"
            correct_res = int(imageName[0])
            image = Image.open(".\\individual_tests\\" + imageName).convert("L")
            # samo kvadratne slike v tem testu
            if image.size[0]  != image.size[1]:
                print(imageName, " ima napacen proporce: ", image.size,". Slika mora biti kvadrat.")
                continue 
            predicted_res = self.partial_img_rec(image, (0,0), (image.size[0], image.size[1]), results=[])

            if predicted_res == []:
                print("Napoved ni mogoča za", imageName)
            else:
                predicted_res = predicted_res[0]

            if predicted_res != correct_res:
                print("napaka v partial_img-rec! Predicted ", predicted_res, ". Pravilen razultat bi bil ", correct_res)
                cnt_wrong += 1
            else:
                cnt_right += 1
                print("pravilno predvideval ",imageName)
        print(cnt_right, " od ", cnt_right + cnt_wrong," števk so bile pravilno zaznane. Stopnja uspeha je zato ", (cnt_right / (cnt_right + cnt_wrong)) * 100," %.")



network = mnist_network()
# preberi stevke in poženi program (iz mape individual_tests)    
network.test_individual_digits()        