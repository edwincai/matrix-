## NaiveBayes.py使用说明
### 项目背景说明
本项目根据C和C++的课程中涉及的知识点，对matrix网站部分题库的题目进行打标签、分类，并按照不同题目的标签重合度进行推荐。
### 代码说明
本代码运用机器学习的方法，采取朴素贝叶斯的算法，以已有题目的题面和标签作为训练集，输入未打标签的题目，预测该题目的标签，整合输出。
### 输入
代码的输入有两方面的数据，一是训练集的数据获取。通过读取json文件，获得题号标签等信息，从本地文件夹读取题目描述，并加入训练集。二是未打标签题目的题面
### 输出
代码的输出为json文件，含所有题目的题号、题目名字、标签等信息，包括训练集的题目题号标签，未打标签的题目题号以及这些题目对应的预测标签。