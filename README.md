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




