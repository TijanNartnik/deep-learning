# Deep Learning - Globoko učenje
## Program za prepoznavo števk iz slik

#### Dodaten Material
- Če vas zanima več o globokem učenju sta tukaj 2 povezavi do zbirke člankov, kjer si lahko podrobno pogledate kako globoko učenje deluje [tukaj](https://www.analyticsvidhya.com/blog/2021/06/a-comprehensive-tutorial-on-deep-learning-part-2/) in [tukaj](https://www.analyticsvidhya.com/blog/2021/05/a-comprehensive-tutorial-on-deep-learning-part-1/)
- Dodaten primer v Pythonu: [tukaj](https://pylessons.com/handwriting-recognition) zaprepoznavo ročno napisanih besed.

  
### Opis
Program z uporabo globoke nevronske mreže prepozna ročno napisane števke iz slik.

Program najprej ustavri in stestira globoko nevronsko mrežo na podatkih MNIST s pomočjo knjižnjice TenserFlow.
Nato pa z zanko pregleda in predvideva katera števka je napisana na vsaki sliki v mapi `individual_tests`. Pomembno je, da so te slike kvadratne oblike, formata `.jpg` ali `.png` in poimenovane z `'stevka'.jpg` (1.jpg) 

#### Uporabljene Knjižnjice
Program je izdelan v pythonu z uporabo knjižnjic:
- TenserFlow: odprtokodna knjižnjica, s fokusom na globoke nevronske mreže in strojno učenje. [spletna stran](teshttps://www.tensorflow.org/t)
- MNIST: Največja baza slik in podatkov ročno napisanih številk, ki se uporablja za treniranje in testiranje različnih modelov strojnega in globokega učenja. [več](https://en.wikipedia.org/wiki/MNIST_database)

#### Video
[Demo Video](demo.mp4)

### Uporaba
Če želite testirati program na svojem računalniku ali ga dodelati oziroma kaj spremeniti:

(python mora biti naložen na računalniku!)

1. Naložite **pip**
Več [tukaj](https://pip.pypa.io/en/stable/installation/).
2. Naložite **TenserFlow**
Več [tukaj](https://www.tensorflow.org/install).
3. Kopirajte ta Github repository v svojo mapo z komando v terminalu (če imate naložen git):
```
git clone https://github.com/TijanNartnik/deep-learning.git
```
4. Poženite program.
   
