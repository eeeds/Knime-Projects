## First Project

[First Project Data ](First_project)

-   Create Table with columns:
    -   `weight`
    -   `height`
    -   `sex`
-   Add Color manager 
-   Add Scatter Plot to visualize the data

![End of First Project](images/first_project.PNG)

## Use Excel Reader for read excel files.
-    Configure it. Make sure that first row is showing the header. 
## Partitioning the data to train and test.
- Use ```Partitioning``` block to partition the data.
- You can set the train and test size with the ```Relative`` and ``Absolute`` options.

## Decision Tree Learner and Decision Tree Predictor
- Use ```DecisionTreeLearner``` block to learn the decision tree.
- Use ```DecisionTreePredictor``` block to predict the data.

![First Model](images/first_model.PNG)

## Data Accessing

[Data Accesing](DataAccessing)

- Create a new workflow called ```DataAccessing```
- Add a new  ```Excel Reader``` block to read the data.
- Add a new  ```File Reader``` block to read the data (csv file).
- Create a table with ```Table Creator``` block.

## Basic Visualization

- Use ```Scatter Plot``` block to visualize the data.
- Use ```Color Manager``` block to color the data.
- Use ```Line Plot``` block to visualize the data in the csv file.
- Use ```Pie Chart``` block to visualize in form of a pie chart.

## Basic Data Manipulation and Preprocessing.

### Row Filtering and Missing Values

-  Use ```Row Filter``` block to filter applying a condition to the data.
- Use ```Rule Based Filter``` block to filter the data based on the rules.
    ```
    $Height$>0 =>  TRUE 
    $Height$<0 =>  FALSE
    $Height$>0 AND $Weight$ >= 50 => TRUE 
    $Height$>0 AND $Weight$ >= 50 AND ($Height$ <220 OR $ID$ >5) => TRUE 
    ```
### Column Filter

- Use ```Column Filter``` block to filter the data based on the column name.

### Concatenate

- Use ```Concatenate ``` block to concatenate data.

![Concatenate](images/concatenate.PNG)

### Joiner

- Use ```Joiner ``` block to join data.

### Group by

- Use ```Groupby``` block to do aggregate functions.
- Group variables using ```Groups``` setting.
- In ```Manual Aggregation``` setting, you can specify the aggregate function.

### String Replacer

- Use ```String Replacer``` block to replace values in string cells if they match a certain wildcard pattern.

### Math Formula

- Use ```Math Formula``` block to apply mathematical formulas to the data.This node evaluates a mathematical expression based onthe values in a row.

![Math Formula](images/math_formula.PNG)

### Auto Binner

- Use ```Auto Binner``` block to bin the data.

### Numeric Binner

- Use ```Numeric Binner``` block to bin the data based on the numeric values.

### Normalizer

- Use ```Normalizer``` block to normalize the data.

### Pivoting

- Use ```Pivoting``` block to pivot the data.
- ```Groups``` setting specify what you want to see in the row header.
- ```Pivot``` setting specify what you want to see in the column header.

### Metanode

Meta nodes are nodes that contain subworkflows, i.e. in the workflow they look like a single node, although they can contain many nodes and even more meta nodes. They are created with the help of the meta node wizard. You can open the meta node wizard by either selecting "Node/Add Meta Node"
from the menu or by clicking the button with the meta node icon in the toolbar (workflow editor must be active).

### Data Generators

Creates random data containing some clusters for Parallel Universes. The data contains a certain fraction of noise patterns and data that is generated to clusters (all clusters have the same size). The data is normalized in [0, 1].

### Column Combiner

- Use ```Column Combiner``` block to combine the data.

### Cell Splitter

- Use ```Cell Splitter``` block to split the data.

### Type conversion

- Use ```String to Number``` block to convert the data.


## Modeling:

Visit [Modeling](https://www.knime.com/nodeguide/analytics/classification-and-predictive-modelling) for more details.

## Classification models:

[Refers to the folder](Classification)

#### Naive Bayes:

- Use ```Naive Bayes Learner``` block to train the model.
        -Limit your nominal values to the ones you want to use (```Nominal Values``` setting).
- Use ```Naive Bayes Predictor``` block to predict the data.

![Naive Bayes Example](images/naive-bayes.PNG)


### Decision Tree:
Decision Tree : Decision tree is the most powerful and popular tool for classification and prediction. A Decision tree is a flowchart-like tree structure, where each internal node denotes a test on an attribute, each branch represents an outcome of the test, and each leaf node (terminal node) holds a class label. 
 
A decision tree for the concept PlayTennis. 
Construction of Decision Tree : 
A tree can be “learned” by splitting the source set into subsets based on an attribute value test. This process is repeated on each derived subset in a recursive manner called recursive partitioning. The recursion is completed when the subset at a node all has the same value of the target variable, or when splitting no longer adds value to the predictions. The construction of a decision tree classifier does not require any domain knowledge or parameter setting, and therefore is appropriate for exploratory knowledge discovery. Decision trees can handle high-dimensional data. In general decision tree classifier has good accuracy. Decision tree induction is a typical inductive approach to learn knowledge on classification. 

- Decision Tree Learner Configuration:
    -   Select the target in the ```Class Column``` setting.
    -   Select the quality measure with ```Quality measure``` setting.
    -   Avoid overfitting using ```Min number records per node``` setting.
    -   You can force the root with ```Force Root split column``` setting.

![Decision Tree](images/decision-tree.PNG)


### K-Nearest Neighbor Algorithm
-   Lazy learner:

        Just store Data set without learning from it

        Start classifying data when it receive Test data

        So it takes less time learning and more time classifying data

-   Eager learner:

        When it receive data set it starts classifying (learning)

        Then it does not wait for test data to learn

        So it takes long time learning and less time classifying data


- Check ```Output class probabilities``` setting to see the probability of the class.

![KNN](images/knn.PNG)


### Distance Metrics

- You can use ```Numeric Distances``` setting to use the Euclidean distance or the Manhattan distance.
- You can use ```K Nearest Neihbor (Distance Function) ``` block to use the previous distance metric.
- Don't forget about normalization before using the distance metric.
- You can use ```Java Distance``` block to use the distance metric in Java.

![Distance Metrics](images/distance-metrics.PNG)

### Support-Vector-Machines

“Support Vector Machine” (SVM) is a supervised machine learning algorithm that can be used for both classification or regression challenges. However,  it is mostly used in classification problems. In the SVM algorithm, we plot each data item as a point in n-dimensional space (where n is a number of features you have) with the value of each feature being the value of a particular coordinate. Then, we perform classification by finding the hyper-plane that differentiates the two classes very well (look at the below snapshot).

![SVM](images/svm.PNG)

### Example of all the models

![Using all models](images/iris-project.PNG)

## Comparing Classification Algorithms
### 1. Problem Type

    All the algorithms considered in here are for the classification. We are going to revisit the algorithms for different problem types in the future.

### 2. Interpretable?

    One major criteria about the machine learning algorithms is the interpretability. Some algorithms are extremely complex and even if they yield the best results, sometimes we do not prefer to use them because it is not easy to understand or explain. So the list of algorithms and their interpretability's:

    Naive Bayes : Sometimes : It is a probabilistic algorithm and remember the theoretical part of our courses. It is explainable but not easy all the time. Especially if you have high number of attributes in your data set it gets more complex to explain.

    Decision Trees: YES: One major advantage of decision trees is the interpretability. I can easily say they are the best algorithms to explain something.

    KNN: YES: it is as simple as explaining the distance between two points. So, if you can find a good result in KNN, you can just visualize the data points on a board, get a ruler and show how they are classified.

    SVM: Sometimes: If the problem is linearly separable, and if you can find a border line between the classes, than you can draw the line between the classes and explain how you find the classification. Unfortunately, it gets more complex with the increasing number of attributes and class labels.

    Logistic Regression: Sometimes: Sometimes it is really easy to understand and explain the outcomes of logistic regression. For example if the problem is binary classification and if you have only 1 attribute, it would be a very easy job to explain. On the other hand, by the increasing number of labels in your classification or by the increasing number of attributes, it gets more and more complex.

### 3. Average Accuracy

    Unfortunately, all the algorithms we have covered until now have low accuracy. In the future classes we are going to cover some techniques like ensemble or neural networks and we are going to talk about why we have higher accuracies.

### 4. Training Speed

    Keep in mind that, these algorithms are very basic and simple algorithms. This brings up a very important speed advantage. So we can say all the algorithms in the list have short training times.

### 5. Prediction Speed

    All the algorithms results the predictions in a short time. Only for the K-NN and if you go with the lazy learning, you might  have difficulty about the prediction time.

### 6. Small number of observations

    KNN: No, requires a certain number of data points, depending on the value of K parameter

    Decision Tree: No, requires some data points to create decision nodes in order to split data set into small trees.

    For the rest of the algorithms: Naive Bayes, SVM or Logistic Regression, you can easily create a decision boundary even if you have small number of data points in your data set.

### 7. Noisy Data

    If a data point is noisy, like we have 250 in height or 300 in weight, the algorithm may be immune to the problem and can handle the situation. All the algorithms except Naive Bayes are weak for the problems like noisy data because they only assume a data point as it is.

    KNN will classify any data point close to the noisy data with the class of noisy data point and will not consider any exception. For the KNN, a noisy data is just another data point and KNN has no immunity against noisy data.

    Decision tree will classify the noisy data as a separate leaf in the tree and any data classification will easily go into the branch of noisy data if it is just close enough. This is also a problem for DT and it can not handle noisy data as a problem and keeps classifying the data points even if they are noisy.

    SVM will just consider the data point in order to find a separation border as any other data points. SVM can not handle the noisy data problem.

    On the other hand the probability of a noisy data in Naive Bayes will be considered as a exception and will not create a problem.

    ### 8. Requires Scaling?

    Only KNN and SVM needs scaling because the parameters are compared with each other during the distance calculation. Don't forget we have normalized the attributes in order to create a better distance metric in KNN. Similarly, SVM also uses the distance and I would strongly recommend you to use scaling / normalization before the SVM algorithm.

    On the other hand, DT, NB or LR never considers axises together. Each axis for the decision tree Naive Bayes or Logistic regression is a completely different story to take care of.

## Association rule learning 
    Association rule learning is a rule-based machine learning method for discovering interesting relations between variables in large databases. It is intended to identify strong rules discovered in databases using some measures of interestingness.[1] In any given transaction with a variety of items, association rules are meant to discover the rules that determine how or why certain items are connected.

    Based on the concept of strong rules, Rakesh Agrawal, Tomasz Imieliński and Arun Swami[2] introduced association rules for discovering regularities between products in large-scale transaction data recorded by point-of-sale (POS) systems in supermarkets. For example, the rule {\displaystyle \{\mathrm {onions,potatoes} \}\Rightarrow \{\mathrm {burger} \}}\{{\mathrm  {onions,potatoes}}\}\Rightarrow \{{\mathrm  {burger}}\} found in the sales data of a supermarket would indicate that if a customer buys onions and potatoes together, they are likely to also buy hamburger meat. Such information can be used as the basis for decisions about marketing activities such as, e.g., promotional pricing or product placements.

    In addition to the above example from market basket analysis association rules are employed today in many application areas including Web usage mining, intrusion detection, continuous production, and bioinformatics. In contrast with sequence mining, association rule learning typically does not consider the order of items either within a transaction or across transactions.

    The association rule algorithm itself consists of various parameters that can make it difficult for those without some expertise in data mining to execute, with many rules that are arduous to understand.[3]

    Despite this, association rule learning is a great system for predicting the behavior in data interconnections. This makes it a noteworthy technique for classification, or discovering patterns in data, when implementing machine learning methods.

    - You can use ```Association Rule Learner ``` to learn association rules from a dataset.

 ## Clustering /Segmentation

    - Centroid Models: K-Means
    - Connectivity Models: Hierarchical Clustering
    - Distribution Based Models
    - Density Models: DBSCAN

### K-Means algorithm

It's an algorithm that works using central points to cluster the data points.
First at all, we need to define the central points (K), then we need to find the closest point to each of the central points.
Then we recalculate the central points again until the central points are the same as the previous ones or until the number of iterations is reached.

### Optimum number of clusters (k value) and WCSS (Within Cluster Sum of Squares)

When you have more k values, the WCSS is going to be lower.

### K-Means in Knime

- You can use ```K-means``` block to applied the algorithm.
- You can set the number of clusters using the ```Number of Clusters``` parameter.

![K-Means](images/k-means.PNG)






