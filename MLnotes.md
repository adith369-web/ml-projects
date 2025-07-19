---
marp: true
theme: default
title: Machine Learning Theory
---
#  Machine Learning Theory
- learning can be supervised, semi-supervised, unsupervised and reinforcement
## supervised learning
- teach computer with examples of known answer
- goal is to create a system to predict the answers for new unseen data based on what it is learned

---
### some basic needed before learning algorithms
- 1. ##### Dataset
    mathematically written as {(x<sub>i</sub>,y<sub>i</sub>)}<sub>i=1</sub><sup>N</sup>
- N is the total number of examples
- (x<sub>i</sub>,y<sub>i</sub>) is an example
x<sub>i</sub> is input or feature vector, y<sub>i</sub> is the output (a known answer)
- 2. #### Feature vector
The **feature vector** (x<sub>i</sub>) is like a list of numbers that describes one example 
it is a vector(mathematically an ordered set of values)


----
the feature vector can have D dimensions.
eg: **bag of word feature vector**
["not spam","miss","prize","money"...."free"] let this be a D dimensional feature vector to check spam email.
these words are unique words collected from n number of emails we used in training
- **condition**: set position to 1 if word appears in email
set zero otherwise(binary feature representation)
a function such that 
if count of 1 > some number:
   show the it as spam 
   else:
    return it as not_spam

----
let the new mail be,"Don't miss this  opportunity to win cash prize"
so the feature vector become [0,1,1,0....,1]
so if the count of 1 is greater that some number model will classify it as spam else not spam
this is a very basic explanation , in models like svm we are not counting the number of "1" instead  using other equation with weights and baises and finally sign function to classify.
**Basic maths**
Basic maths  like vector,set,addition,statistics,probability,Matrix properties are also needed



---
# Machine Learning Algorithms
### <h2 style="text-decoration: underline;">general working principle until chapter 4</h2>
in supervised learning there will be an equation which contain weight and bais, there will another equation for called **loss function**, the loss function calculates error between predicted values (by the  formuala with some arbitary w and b) and actual values given in training data set, goal of training is to minimize the loss by adjusting model parameters (weight and bias)
thus we will get the best w and b.
if parameter is adjusted too strictly for all dataset it causes **overfitting**,in opposite way it causes **underfitting**.

---
## Support Vector Machine(SVM)
- svm see's every feature vector as a point in high-demsional space
- As we already learned bag of words method, upto vectorization the steps are same.
- the main goal is to finds the best  decision boundary that seperates the data classes with maximum length.
- hyperplane equation is w.x-b=0
w(weight),b(bias),x is feature vector.
---
![Alt text](https://media.licdn.com/dms/image/v2/D4D12AQHp3hPudcf1Mw/article-cover_image-shrink_600_2000/article-cover_image-shrink_600_2000/0/1693156828175?e=2147483647&v=beta&t=djr7Q_Y8rjH9c34bjjXpQ7YfuZtLTKwUXfSPqGgzGVQ)
that middile line is the line with equation wx+b=0 is hyperplane
**the hyper plane seperates the data in to two classes:**
- for class 1 : w.x+b &ge; 1
- for class -1 : w.x+b &le; -1
- these two equations define decision boundary
---
**Key Characteristics of w and b:**
- Determines the direction of the hyperplane. 
- Does not specify the exact position of the hyperplane; the bias (b) does that.
- Its magnitude (||w||) impacts the margin of the hyperplane. To maximize margin, SVM minimizes ||w||, which makes the hyperplane more robust.
ie distance of a point from hyperplane is determined by w and b determine the distance hyperplane is from origin.
note: w is vector and b is number
---
**Margins**
The margin is the distance between the hyperplane and the nearest data points of each class (support vectors). It's defined as:
<span style="font-size:1.2em;">margin = <sup>2</sup>&frasl;<sub>||w||</sub></span>
The goal of SVM is to maximize the margin, which ensures the hyperplane is as far away as possible from the closest data points.
we can derive it by directly computing the perpendicular distance between the two support hyperplanes (parallel) using vector subtraction. 
- so to maximize <span style="font-size:1.2em;">margin = <sup>2</sup>&frasl;<sub>||w||</sub></span>
we need to minimize ||w|| but since it is a vector with magnitude <span style="font-size:1.2em;">√(x<sup>2</sup> + y<sup>2</sup> + ...)</span>
---
its better to minimize <span style="font-size:1.2em;">∥w∥<sup>2</sup>&frasl;2</span> for easy calculation.
**Weight Vector (𝑤⃗ )**

The weight vector in SVM is given by:

<span style="font-size:1.2em;">
𝑤⃗ = ∑<sub>i=1</sub><sup>N</sup> α<sub>i</sub> y<sub>i</sub> 𝑥⃗<sub>i</sub>
</span>

Where:
- α<sub>i</sub>: Lagrange multipliers (learned during optimization).
- y<sub>i</sub>: Class label (+1 or −1).
- 𝑥⃗<sub>i</sub>: Support vectors (training points where α<sub>i</sub> &gt; 0).
- α<sub>i</sub> controls the influence of each training point on the final SVM model.

---
Specialized algorithms (like Sequential Minimal Optimization, or SMO) are used to find the values of α<sub>i</sub> that maximize the dual objective function(a complex function) under the constraints.
**Bias Term**

The bias term in SVM is given by:

<span style="font-size:1.2em;">
b = <sup>1</sup>&frasl;<sub>2</sub> ( min<sub>i: y<sub>i</sub>=1</sub> (𝑤⃗ ⋅ 𝑥⃗<sub>i</sub>) + max<sub>i: y<sub>i</sub>=−1</sub> (𝑤⃗ ⋅ 𝑥⃗<sub>i</sub>) )
</span>

- First term: Smallest output for positive-class support vectors.
- Second term: Largest output for negative-class support vectors.

---
# Challenges in SVM and Solutions

1. Linear Separability
Problem: 
SVM assumes data is linearly separable, but real-world data often isn't.
Solution:  
Kernel Trick:Map data to higher dimensions using kernels (e.g., RBF, polynomial).
Example: A circle in 2D becomes separable as a plane in 3D.
---

2. Noise and Overfitting
Problem:  
Outliers or mislabeled data can skew the hyperplane.
solution:  
- Soft-Margin SVM: Introduce slack variables (ξ<sub>i</sub>) to allow some misclassifications.
---
### linear regression
- like svm it also consider feature vectros as points in higher dimension space but here all points contribute equally not just support vectors
- here instaed of seperating the points into two classes the hyperplane Predict a continuous value y by minimizing the vertical distance (error) between the hyperplane and all data points.
The equation for the hyperplane in linear regression is:
        <span style="font-size:1.2em;">
        y = w<sup>T</sup>x + b
        </span>

---
Where:
- **w**: Weight vector (slopes for each feature)
- **b**: Bias (y-intercept)
**optimization**
the goal is to find or adjust correct values for w, b such that the error or diviation from actual value y is minimum
The **Mean Squared Error (MSE)** is defined as:

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N}(y_i - \hat{y}_i)^2$$
---
- $y_i$ = actual (observed) value
- $\hat{y}_i$ = predicted value ($\hat{y}_i = \mathbf{w}^T\mathbf{x}_i + b$)
- $N$ = number of data points

To find the best parameters $(\mathbf{w}, b)$, we minimize the MSE:

$$\min_{\mathbf{w},b} \sum_{i=1}^{N}(y_i - (\mathbf{w}^T\mathbf{x}_i + b))^2$$

Why Squared Error?
Differentiability
The square function is smooth, making optimization easier (gradient descent)

---
Penalizes Large Errors
Squaring amplifies large deviations 

**Logistic Regression**
it is a classification algoritm despite having regression in name.
solves binary classification by mapping linear outputs(-&infin;, +&infin;) to probabilites (0-1) using sigmoid function.
The sigmoid function in logistic regression is:
$$f(x) = \frac{1}{1 + e^{-(w^T x + b)}}$$
eg: if w.x + b = 2 then f(x)&asymp;.88 that is 88% probability

---
to covert probability into classes eg:(0,1), we apply a threshold (usually 0.5 but we can change accordinglr):
eg:
if f(x) &ge; 0.5,predict class 1
if f(x) &le; 0.5, predict class 0
**loss function**
$$
L = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i) \right]
$$
optimization method is same as in linear regression (gradient descent)
- Compute the gradient of the loss with respect to **w** and **b**

---
- Update rule for each iteration:
  - w := w - η · ∂L/∂w
  - b := b - η · ∂L/∂b
  
  Repeat until convergence (loss stops decreasing).

**Decision Tree Algorithm**
a supervised Ml algorithm for classification and regression
here ID3 algorithm is using which is only for classification.
a decision tree is like a flow chart like struture used for classification and regression, works recursively splitting the dataset into subsets based on feature  values to form a tree structure.

---
<img src="https://pub.mdpi-res.com/algorithms/algorithms-10-00124/article_deploy/html/images/algorithms-10-00124-g001.png?1569644990" width="50%">

internal nodes represent atrributes
branches represent decisions or tests on attribute
leaf node represents clas labels(output)

---
**steps in decision tree algorithm**
1. start with entire dataset as root
2. select the best atrribute tp split the data using a metric (eg:information gain in ID3)
- to select the best attribute we need to calculate the information gain of each attribute and the attribute with highest value will the best attribute.
3. split dataset into subset, based on the seletec attribute
4. repeat the process recursively for each subset until no more attributes remian or max depth reached.

---
**how to select best atrribute?**
first we need to calculate entropy of entire dataset
 $$
\text{Entropy}(S) = -\sum_{i} p_i \log_2(p_i)
$$
pi= p1,p2 ..., this represent the probability of class to occur
eg:p1=p(pass),p2=p(fail)
secondly we need to calculate entropy of each attribute
here pi=p(class) if i is happening
eg: let atrribute be studyhours
pi=p(pass/high study hours),p(fail/highstudyhours)
like that we need to find entropy for study hours high, low, medium etc.

---
then we need to calculate the total entropy of that particular attribute we selected.
for that the formula is:
$$
\text{Entropy}(\text{attribute}) = \sum_{v \in \text{Values}} \frac{|S_v|}{|S|} \cdot \text{Entropy}(S_v)
$$
where  
- \(S_v\): subset of data where attribute has value \(v\)  
- \(|S_v|\): number of samples in \(S_v\)  
- \(|S|\): total number of samples  

---
the best attribute is with which have highest information gain value:
information gain of a particular attribute = entropy of full dataset - entropy of that attribute
so , we will get the best attribute and we will start splitting from that attribute, then we will get stuck in a value then again we repeat to process recursively.

**KNN(K-Nearest Neighbour Algorithm)**
K-Nearest Neighbour is one of the simplest Machine Learning algorithms based on Supervised Learning technique
K-NN algorithm assumes the similarity between the new case/data and available cases and put the new case into the category that is most similar to the available categories.

---
it does not learn from the training set immediately instead it stores the dataset and at the time of classification, it performs an action on the dataset.
**how does KNN works**?
<img src="https://miro.medium.com/v2/resize:fit:1400/0*34SajbTO2C5Lvigs.png" width="400" height="400">

---

• Step-1: Select the number K of the neighbours (we can decide the K)  

• Step-2: Calculate the Euclidean distance of K number of neighbours  

• Step-3: Take the k nearest neighbours as per the calculated Euclidean distance.  

• Step-4: Among these k neighbours, count the number of the data points in each category.  

• Step-5: Assign the new data points to that category for which the number of the neighbour is maximum.  

• Step-6: Our model is ready.  

note: Large values for K are good, better to be atleast greater than 5

---
**Feature Engineering**

the problem of transforming row data into dataset is called feature engineering.
so we need datasets with highly informative features, highly informative features are also called features with high predictive power.

eg-average time session have higher predictive power for predicting whether the user will use the app in future.

We say that a model has a low bias when it predicts well the training data.(it has no connection with same like bais term).

