# Propagacion de etiquetas: implementacion y comparativa

## Objetivo

Este repo incluye un ejemplo sencillo de aprendizaje semisupervisado con el algoritmo
**Label Propagation**. La implementacion esta en:

- `notebooks/label_propagation_semisupervisado.ipynb`

El notebook fue ejecutado completo y deja salidas guardadas. Usa un dataset sintetico
`make_moons`, oculta casi todas las etiquetas del conjunto de entrenamiento y compara:

- un baseline supervisado (`KNN`) entrenado solo con pocas etiquetas reales;
- `LabelPropagation`, basado en propagacion sobre un grafo de similitud;
- `LabelSpreading`, variante regularizada relacionada con consistencia local y global.

## Fuentes revisadas

1. Zhu, X., Ghahramani, Z. y Lafferty, J. (2003). *Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions*. ICML.  
   Fuente consultada: https://mlanthology.org/icml/2003/zhu2003icml-semi/

2. Zhou, D., Bousquet, O., Lal, T. N., Weston, J. y Scholkopf, B. (2004). *Learning with Local and Global Consistency*. NIPS 16.  
   Fuente consultada: https://is.mpg.de/publications/2333

3. van Engelen, J. E. y Hoos, H. H. (2020). *A survey on semi-supervised learning*. Machine Learning, 109, 373-440.  
   Fuente consultada: https://link.springer.com/article/10.1007/s10994-019-05855-6

## Implementacion realizada

El notebook genera 500 observaciones de dos clases no lineales con `make_moons`. Luego separa
325 observaciones para entrenamiento y 175 para prueba. En entrenamiento solo conserva 8
etiquetas reales, 4 por clase, y marca las otras 317 observaciones como no etiquetadas usando
`-1`, que es el formato esperado por `sklearn.semi_supervised`.

El flujo fue:

1. Crear datos no lineales con dos clases.
2. Dejar solo 2.46% de etiquetas visibles en entrenamiento.
3. Entrenar un KNN solo con esas 8 etiquetas.
4. Entrenar `LabelPropagation` usando las 8 etiquetas y los 317 puntos sin etiqueta.
5. Entrenar `LabelSpreading` como variante regularizada.
6. Comparar accuracy en test, matriz de confusion y etiquetas propagadas.
7. Evaluar sensibilidad del grafo cambiando `gamma` en el kernel RBF.

## Resultados del notebook ejecutado

| Modelo | Accuracy test | Errores test |
|---|---:|---:|
| LabelPropagation_RBF | 0.994286 | 1 |
| LabelSpreading_RBF | 0.988571 | 2 |
| KNN_con_pocas_etiquetas | 0.845714 | 27 |

El mejor modelo fue `LabelPropagation_RBF`. Su matriz de confusion fue:

| | Pred 0 | Pred 1 |
|---|---:|---:|
| Real 0 | 88 | 0 |
| Real 1 | 1 | 86 |

En los 317 puntos de entrenamiento originalmente sin etiqueta, `LabelPropagation` logro
0.996845 de accuracy al comparar las etiquetas propagadas contra las clases reales ocultas.

## Comparativa con los hallazgos de las fuentes

### Zhu, Ghahramani y Lafferty (2003)

La fuente plantea que los datos etiquetados y no etiquetados pueden representarse como vertices
de un grafo, donde las aristas codifican similitud entre instancias. El notebook aplica esa idea
con un kernel RBF: puntos cercanos en la forma de las lunas reciben pesos altos y la etiqueta se
propaga por esa estructura.

Hallazgo comparable: con solo 8 etiquetas reales, `LabelPropagation` supero claramente al KNN
supervisado. Esto muestra que los puntos no etiquetados si aportaron informacion util cuando el
grafo reflejo bien la geometria de los datos.

### Zhou et al. (2004)

Zhou et al. enfatizan una funcion de clasificacion suave respecto a la estructura intrinseca
revelada por puntos etiquetados y no etiquetados. En el notebook, `LabelSpreading` representa esa
familia de ideas: busca consistencia local entre vecinos, pero mantiene una regularizacion global
para no depender de una propagacion completamente rigida.

Hallazgo comparable: `LabelSpreading` tambien mejoro mucho frente al baseline supervisado
(0.988571 contra 0.845714). Quedo apenas debajo de `LabelPropagation` en esta corrida, pero con
solo 2 errores de test. Esto confirma que la suavidad sobre la estructura de datos puede funcionar
bien aun con pocas etiquetas.

### van Engelen y Hoos (2020)

La encuesta explica que el aprendizaje semisupervisado es especialmente relevante cuando las
etiquetas son escasas y los datos sin etiqueta pueden aportar informacion bajo supuestos sobre la
distribucion. Tambien resume que los metodos basados en grafos son intuitivos, pero sensibles a
la construccion del grafo y con retos computacionales.

Hallazgo comparable: la sensibilidad a `gamma` mostro que el resultado no es automatico.

| gamma | Accuracy test | Accuracy propagada train sin etiqueta |
|---:|---:|---:|
| 1 | 0.525714 | 0.517350 |
| 5 | 0.880000 | 0.883281 |
| 10 | 0.937143 | 0.924290 |
| 20 | 0.994286 | 0.996845 |
| 35 | 1.000000 | 1.000000 |
| 60 | 1.000000 | 1.000000 |

Con `gamma=1`, el grafo suaviza demasiado y casi pierde la estructura de clases. Con valores
mayores, el grafo captura mejor los vecindarios locales de las dos lunas. Esto conecta con la
advertencia de la encuesta: los datos sin etiqueta ayudan si la estructura de la distribucion
esta alineada con las clases.

## Conclusion

La implementacion confirma el valor practico de la propagacion de etiquetas en un escenario con
pocas etiquetas. En este caso, usar los puntos sin etiqueta redujo los errores de test de 27 a 1
frente al baseline supervisado. Tambien muestra una limitacion importante: el rendimiento depende
de construir un grafo de similitud adecuado. Por eso, en aplicaciones reales, no basta con aplicar
Label Propagation; tambien hay que revisar escalado de variables, metrica de similitud, parametros
del grafo y si las clases respetan una estructura de vecindad clara.
