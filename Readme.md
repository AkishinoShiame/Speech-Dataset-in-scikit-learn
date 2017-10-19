# Speech-Dataset-in-scikit-learn

This project is the after-work of Graduated works.
Testing the Datasets of [Chinese-Speech-Emotion-Datasets](https://github.com/artmusic0/Chinese-Speech-Emotion-Datasets).

## Details

| Method     | DataSet | TrainTime      | Accuracy(Train) | Accuracy(Test) |
| ----------:|:------- |:--------------:|:---------------:|:-------------- |
| Bayes      | Gray    | 1.028 sec      | 96.5%           | 94.23%         |
| DisionTree | Gray    | 3.518 sec      | 100.0%          | 100%           |
| SVM        | Gray    | 135.714 sec    | 100.0%          | 38.46%         |
| K-NN       | Gray    | 1.441 sec      | 99.83%          | 100%           |
| NN         | Gray    | 33.084 sec     | 38.0%           | 40.38%         |
| LeNet      | Gray    | 10 hour 17 min | 98%             | 100%           |
| AlexNet    | Gray    | 6 hour 14 min  | 95%             | 96.15%         |
| GoogLeNet  | Gray    | 1 hour 20 min  | 96.5%           | 100%           |

| Method     | DataSet | TrainTime     | Accuracy(Train) | Accuracy(Test) |
| ----------:|:------- |:-------------:|:---------------:|:-------------- |
| Bayes      | Color   | 4.763 sec     | 97%             | 94.23%         |
| DisionTree | Color   | 9.550 sec     | 100%            | 100%           |
| SVM        | Color   | 43.221 sec    | 33.33%          | 34.62%         |
| K-NN       | Color   | 3.931 sec     | 99.67%          | 100%           |
| NN         | Color   | 218.309 sec   | 100.0%          | 100%           |
| LeNet      | Color   | 1 hour 23 min | 98%             | 100%           |
| AlexNet    | Color   | 5 hour 38 min | 95%             | 96.15%         |
| GoogLeNet  | Color   | 1 hour 22 min | 98%             | 100%           |

## Important !!!

Please cites this peper belows if you used it.

Lee, M. C., Yeh, S. C., Chiu, S. Y., & Chang, J. W. (2017, June). [A Deep Convolutional Neural Network Based Virtual Elderly Companion Agent.](http://dl.acm.org/citation.cfm?id=3083220)

```
Ming Che Lee, Sheng Cheng Yeh, Sheng Yu Chiu, Jia Wei Chang. "A Deep Convolutional Neural Network Based Virtual Elderly Companion Agent", MMSys'17 Proceedings of the 8th ACM on Multimedia Systems Conference, Pages 235-238, June 20 - 23, 2017, Taipei, Taiwan. (EI, SIGMM, accept rate: 28%)
```
