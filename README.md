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





