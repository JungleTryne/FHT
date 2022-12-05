# Fast Hough Transform

Данила Мишин, Б05-922

Запуск
```bash
python3 main.py --image_path ./examples/1.jpg --output_path ./examples_result_bilinear/1.jpg
```

```bash
python3 main.py --help

Usage: main.py [OPTIONS]

Options:
  --image_path PATH       Path to the the image  [required]
  --output_path PATH      Path to the resulting image.         If not
                          specified, the image will be shown in a pop-up
                          window.
  --interpolation TEXT    [Internal] Interpolation type used to get the angle
                          (bilinear - default). Options: bilinear, neighbour
  --show_hough_transform  Show result of Hough transform
  --help                  Show this message and exit.

```

Запуск бенчмарка
```bash
python3 benchmark.py --images_path ./examples
```

Асимптотика алгоритма: `O(mnlogn)`, 
где `m` - высота, `n` - ширина изображения, при условии, что они равны
степени двойки. Если нет, то они округляются вверх путем добавления нулей
справа и снизу, то есть тогда будет асимптотика
`O( m_0 * n_0 * log2 (n_0) )`, где `m_0 = 2 ** ceil(log2(m))`,
`n_0 = 2 ** ceil(log2(n))`.

По памяти: во время стадии merge мы создаем новый массив нулей
и заполняем его. Поэтому требование по памяти: `O(n_0 * m_0)`

Перед применением алгоритма изображение подлежит предобработке:
- Становится черно-белым
- Производится морфологическое замыкание с ядром размера 5x5
- Производится нахождение границ с помощью оператора Кэнни
И лишь затем запускается алгоритм


Результат бенчмарка: `benchmark.png`
Результат работы переворачивания изображения лежит в папках
`examples_result_bilinear` и `examples_result_neighbour`,
в зависимости от способа интерполяции при нахождении угла наклона.


