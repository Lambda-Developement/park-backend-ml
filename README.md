# Пример построения предсказания

1. Выберите парковку.
2. Вручную разметьте её парковочные места  с помощью LabelImg, используя лейбл parking lot, и сохраните в формате .json (CreateML)
![img](example/step_1_label.png)


3. `python3 extra_scripts/show_predict.py example/Парковка_РИО_2022_09_07.json example/Парковка_РИО_2022_09_07.png`, чтобы получить занятость парковки на картинке.

```bash
[quakumei@boxpc park-backend-ml]$ python3 extra_scripts/show_predict.py example/Парковка_РИО_2022_09_07.json example/Парковка_РИО_2022_09_07.png
[False, False, False, False, False, False, False, False, True, True, True, True, False, True, False, True, False, False, False, False, True, True, False, False]
Occupancy rate of parking lot on example/Парковка_РИО_2022_09_07.png is 0.3333333333333333 (8/24)
```
