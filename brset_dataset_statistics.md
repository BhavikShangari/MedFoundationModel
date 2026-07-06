# BRSET Dataset Distribution Statistics

Disease category definitions:

- no_disease: disease_count == 0
- single_disease: disease_count == 1
- multiple_disease: disease_count >= 2

## Overview

- Total images: 16266
- Unique patients: 8524
- Disease indicator columns: 13
- Test images per category: 50
- Patient overlap between retrieval and test: 0

## Split Summary

| split     | images | patients | percent_images |
| --------- | ------ | -------- | -------------- |
| retrieval | 16116  | 8449     | 99.08          |
| test      | 150    | 75       | 0.92           |

## Disease Category By Split

| split     | no_disease | single_disease | multiple_disease | total |
| --------- | ---------- | -------------- | ---------------- | ----- |
| retrieval | 8411       | 5931           | 1774             | 16116 |
| test      | 50         | 50             | 50               | 150   |

## Disease Count By Split

| split     | 0    | 1    | 2    | 3   | total |
| --------- | ---- | ---- | ---- | --- | ----- |
| retrieval | 8411 | 5931 | 1602 | 172 | 16116 |
| test      | 50   | 50   | 45   | 5   | 150   |

## Overall Disease Category Distribution

| disease_category | count | percent |
| ---------------- | ----- | ------- |
| multiple_disease | 1824  | 11.21   |
| no_disease       | 8461  | 52.02   |
| single_disease   | 5981  | 36.77   |

## Overall Disease Count Distribution

| disease_count | count | percent |
| ------------- | ----- | ------- |
| 0             | 8461  | 52.02   |
| 1             | 5981  | 36.77   |
| 2             | 1647  | 10.13   |
| 3             | 177   | 1.09    |

## Disease Label Prevalence: Entire Dataset

| disease                  | positive_images | percent_of_images |
| ------------------------ | --------------- | ----------------- |
| increased_cup_disc       | 3205            | 19.7              |
| drusens                  | 2833            | 17.42             |
| diabetic_retinopathy     | 1070            | 6.58              |
| other                    | 820             | 5.04              |
| macular_edema            | 401             | 2.47              |
| amd                      | 299             | 1.84              |
| scar                     | 291             | 1.79              |
| hypertensive_retinopathy | 284             | 1.75              |
| myopic_fundus            | 270             | 1.66              |
| nevus                    | 130             | 0.8               |
| vascular_occlusion       | 101             | 0.62              |
| hemorrhage               | 95              | 0.58              |
| retinal_detachment       | 7               | 0.04              |

## Disease Label Prevalence: Retrieval Split

| disease                  | positive_images | percent_of_images |
| ------------------------ | --------------- | ----------------- |
| increased_cup_disc       | 3160            | 19.61             |
| drusens                  | 2780            | 17.25             |
| diabetic_retinopathy     | 1054            | 6.54              |
| other                    | 809             | 5.02              |
| macular_edema            | 395             | 2.45              |
| amd                      | 291             | 1.81              |
| scar                     | 290             | 1.8               |
| hypertensive_retinopathy | 276             | 1.71              |
| myopic_fundus            | 265             | 1.64              |
| nevus                    | 130             | 0.81              |
| vascular_occlusion       | 101             | 0.63              |
| hemorrhage               | 93              | 0.58              |
| retinal_detachment       | 7               | 0.04              |

## Disease Label Prevalence: Test Split

| disease                  | positive_images | percent_of_images |
| ------------------------ | --------------- | ----------------- |
| drusens                  | 53              | 35.33             |
| increased_cup_disc       | 45              | 30.0              |
| diabetic_retinopathy     | 16              | 10.67             |
| other                    | 11              | 7.33              |
| amd                      | 8               | 5.33              |
| hypertensive_retinopathy | 8               | 5.33              |
| macular_edema            | 6               | 4.0               |
| myopic_fundus            | 5               | 3.33              |
| hemorrhage               | 2               | 1.33              |
| scar                     | 1               | 0.67              |
| nevus                    | 0               | 0.0               |
| vascular_occlusion       | 0               | 0.0               |
| retinal_detachment       | 0               | 0.0               |

## Images Per Patient

| images | count | percent |
| ------ | ----- | ------- |
| 1      | 804   | 9.43    |
| 2      | 7706  | 90.4    |
| 3      | 6     | 0.07    |
| 4      | 8     | 0.09    |

## Patient Age Summary

| statistic   | value   |
| ----------- | ------- |
| non_missing | 10820.0 |
| missing     | 5446.0  |
| mean        | 57.66   |
| median      | 61.0    |
| min         | 5.0     |
| max         | 97.0    |

## patient_sex Distribution

| patient_sex | count | percent |
| ----------- | ----- | ------- |
| 1           | 6214  | 38.2    |
| 2           | 10052 | 61.8    |

## exam_eye Distribution

| exam_eye | count | percent |
| -------- | ----- | ------- |
| 1        | 8155  | 50.14   |
| 2        | 8111  | 49.86   |

## camera Distribution

| camera       | count | percent |
| ------------ | ----- | ------- |
| Canon CR     | 10591 | 65.11   |
| NIKON NF5050 | 5675  | 34.89   |

## quality Distribution

| quality    | count | percent |
| ---------- | ----- | ------- |
| Adequate   | 14280 | 87.79   |
| Inadequate | 1986  | 12.21   |

## diabetes Distribution

| diabetes | count | percent |
| -------- | ----- | ------- |
| No       | 13687 | 84.14   |
| yes      | 2579  | 15.86   |

## nationality Distribution

| nationality | count | percent |
| ----------- | ----- | ------- |
| Brazil      | 16266 | 100.0   |
