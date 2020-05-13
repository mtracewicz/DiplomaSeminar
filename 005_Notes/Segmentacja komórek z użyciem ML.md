# Segmentacja komórek z użyciem ML

## "Automated Training of Deep Convolutional Neural Networks for Cell Segmentation"

[nature - article](https://www.nature.com/articles/s41598-017-07599-6)

Artykuł opisuje wykorzystanie głębokich sieci splotowych do segmentacji komórek. Pokazuje on, że jeżeli mamy obrazy zawierające fluorescencyjnie  zaznaczone jądra i cytoplazmy to z pomocą preprocessingu i sieci splotowej jesteśmy w stanie dokonać segmentacji. Artykuł ilustruje bardziej fakt, że się da niż konkretną merytoryczną wiedzę jak to zrobić.

## "A deep learning-based algorithm for 2-D cell segmentation in microscopy images"

[biomedcentral - article](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2375-z)

Artykuł na temat segmentacji komórek na obrazach z jednym kanałem przy użyciu głębokich sieci neuronowych.
Osiągnięta dokładność 86%. Artykuł skupia się na rozpoznawaniu całych komórek na obrazach 2D, gdzie tło jest czarne a komórki jasne. Segmentacja składa się z trzech etapów:

1. Znalezienie komórek

2. Oddzielenie komórek stykających się

3. Podział elementów sub-komórkowych

   W przeciwieństwie do segmentacji jąder komórkowych separacja całych komórek jest bardziej skomplikowana ponieważ cytoplazma w przeciwieństwie do jąder przyjmuje najróżniejsze kształty i rozmiary. Co więcej stykające się  komórki mogą mieć bardzo słabo stykające się granice.

   Istnieje stosunkowo wiele algorytmów pozwalających na segmentację jąder komórkowych na zdjęciach 2D.
   Istnieją także, jednak zdecydowanie mniej liczne algorytmy pozwalające nam na segmentację całych komórek. Jednak algorytmy te działają tylko w sytuacji gdy komórki na, których zachodzi operacja są podobnych kształtów i rozmiarów. Ostatnimi czasy pojawiły się również próby użycia głębokiego uczenia maszynowego w celu segmentacji. Próby te były podejmowane na obrazach jedno i dwu kanałowych. Jednak nie było sytuacji w, której pomyślnie udało by się rozdzielić granice komórek. Próby przy użyciu sieci splotowych miały lepsze wyniki, jednak nie udało się oddzielić komórek stykających się. Przez rosnące zainteresowanie zostały zorganizowane trzy konkursy polegające na śledzeniu komórek w serii zdjęć. Użycie różnych markerów, różnych poziomów zbliżenia oraz różnego rodzaju komórek powoduje duże zróżnicowanie danych oraz znacznie utrudnia zadanie. Powyższe utrudnienia sprawiają, że utworzenie algorytmu generalnego jest bardzo wymagające oraz potrzebne. 

   Aby ujednolicić działanie sieci obraz wejściowy najpierw ma wygaszone tło za pomocą filtra 200x200px a następnie pixele są przekształcane aby przybliżenie było około dziesięciu krotne. Sieć była napisana z użyciem biblioteki MXNET i używa architektury typu UNET. Sieć w pierwszym kroku kroku nadaje fragmentom 160x160px etykiety jądro, cytoplazma, tło. Następnie obrazy przepuszczane są przez pięć par warstw splotowych plus zmniejszających. Rozmiar filtrów to 3x3 a ich ilość wynosi w kolejnych warstwach kolejno 32, 64, 128, 128, 256. Dodatkowo sieć posiada trzy warstwy odrzucające w celu zmniejszenia zbytniego dopasowania. Jako funkcja strat została użyta funkcja root-mean-square-deviation (RMSD). Sieć w procesie uczenia wykonuje 30 epochów. W pojedynczym przebiegu sieć przetwarzała trzydzieści dwa obrazy, a parametr learning rate został ustalony na 0.001. W wyniku przejścia przez sieć neuronową tworzona jest mapa (nowy obraz) na której piksele pokazują prawdopodobieństwo, że w danym miejscu jest jądro, cytoplazma, tło. Na podstawie tych map jesteśmy w stanie dokonać segmentacji.

   Proces uczenia sieci trwał około sześciu godzin. Wynikowa sieć potrafi dokonać predykcji w czasie 4-6 minut.

## Cell proposal network

[github](https://github.com/SaadUllahAkram/cpn)

Program napisany w MATLAB z użyciem frameworka Caffe, który proponuje maski dla komórek z obrazów mikroskopowych. Używane do śledzenia komórek. 

## Cell segmentation in video

[github](https://github.com/iitmcvg/Cell-Segmentation)

System pozwalający na segmentację komórek na wideo przy użyciu metody "Structured Forest". Napisany w języku Python wykorzystuje biblioteki OpenCV i NumPy

## Cellpose - algorytm !! (Najbardziej obiecujące)

[github](https://github.com/MouseLand/cellpose)

Algorytm do segmentacji komórek i jąder komórkowych. Napisany w Pythone z użyciem scipy i numpy. Posiada własne GUI oraz jest dostępne w okrojonej wersji w [przeglądarce](http://www.cellpose.org/). Posiada minimalne wymaganie 8GB pamięci RAM (zalecane 16-32 do dużych obrazów). Aby otworzyć przykład użycia jako jupyter notebook klikamy [tu](https://colab.research.google.com/github/MouseLand/cellpose/blob/master/notebooks/run_cellpose.ipynb).

## Deep learning 64x64 monochrome images (Wydaje się podobny do tego co miałbym robić)

[github](https://github.com/bomri/deep_cell_segmentation)

Projekt pozwalający na segmentację czarno-białych obrazów. Można powiedzieć, że wyostrza na obrazku komórki. Przygotowany do działania na obrazach mikroskopowych dla komórek rakowych raka płuc. Napisane w Pythone z użyciem Tensorflow. Architektura: sześć warstw splotowych (16, 32, 32, 64, 64, x filtrów, gdzie x jest zmienne zależne od rozmiaru danych wejściowych na, których model był uczony.)

## Cell recognition

[github](https://github.com/UmutSahinSE/CellRecognition)

Skrypt pozwalający na segmentację komórek na podstawie obrazów typu "Phase-contrast". Jest napisany z użyciem Pythona i numpy, projekt akademicki. Z tego co mi się wydaje nie jest tu stosowane uczenie maszynowe a jedynie algorytm iteracyjny. Najpierw znajdowane są krawędzie na obrazie, potem stosowane jest wygładzanie i redukcja tła. Nakładane są na siebie obraz utworzony w ten sposób oraz obraz wskazujący obiekty stacjonarne (?). Następnie na obrazie wynikowym bliskie krawędzie są łączone, i wypełniane a małe obiekty usuwane.

