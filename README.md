# AtlasNet2

## Постановка задачи

Есть проект AtlasNet (https://github.com/ThibaultGROUEIX/AtlasNet), в нём на датасете ShapeNet делается автокодирование
моделей, а так же реконструкция 3D модели по одному изображению. В модели автокодировщика в качестве энкодера 
используется энкодер от PointNet. Проблема этой модели в том, что она терияет мелкие детали при автокодировании.

Задача -- заменить энкодер от PointNet на энкодер от PointNet++. С высокой вероятностью это улучшит результаты на 
ShapeNet и добавит больше детализации результату. Из моей работы достоверно известно, что подобный трюк улучшает 
результаты автокодирования на датасете wax up'ов (внешняя поверхность модели зубной коронки) коронок для 36 (по FDI) 
зубов, как по метрике Chamfer, так и визуально в детализации.


## Про статью

В статью предполагается добавить помимо результатов на ShapeNet результаты на закрытом датасете wax up'ов зубных коронок
для 36 зубов со скриншотами, демонстрирующими результат, но без публикации самого датасета. Нужно это для того, чтобы 
продемонстрировать то, что полученная штука работает не только на ShapeNet, но и на других датасетах.

Руководство от меня требует, чтобы в статье было указано в какой я компании работаю, а при демонстрации результатов на 
датасете wax up'ов было указано, что он предоставлен компанией Glidewell Labs.


## Возможное дополнение задачи

Возможно, стоит попробовать заменить энкодер AtlasNet на энкодер других, отличных от PointNet/PointNet++ сетей. Или это,
возможно, тема для других статей?


## Распределение обязанностей в команде

Предлагаю следующее распределение ролей в команде:

    * Кудин Степан -- разработка кода, проведение эксперементов, помощь по написанию статьи.
    * Нотченко Александр -- написание статьи, публикация в нормальном журнале.
