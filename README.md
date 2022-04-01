# Machine-Learning-Roadmap
Machine Learning is a process of converting real-life scenarios into mathematical models consisting of numbers and then finding patterns in those numbers to be able to predict behavior or perform actions that are analogous to that real-world scenario.

## Difference between Traditional Programming and Machine Learning
### Traditional Programming
In traditional programming, we analyze the problem and break it down into simple rules that we hard code in the program and that program then takes in input and processes it using our prescribed rules to provide output.
### Machine Learning
Machine learning is a bit different from traditional programming as we do not hard code any rules into the program due to fundamental limitations but we write an algorithm that is smart enough to figure out the rules on its own, given a set of inputs and desired outputs.

## Why Machine Learning?
A good reason to use machine learning is that we **can** use it, cause why not?... But an even better reason is that some problems are so complex that it's not feasible for us to figure out all the rules, in these scenarios machine learning is not only the easiest but perhaps the most optimal solution. This does not suggest that we should always blindly use machine learning, cause if you can build a simple rule-based system that doesn’t require machine learning, do that instead.

## Strengths of Machine Learning
- **Problems with long lists of rules:** when the traditional approach fails, machine learning may help.
- **Continually changing environments:** machine learning can adapt (‘learn’) to new scenarios.
- **Discovering insights within large collections of data:** can you imagine trying to go through every transaction your (large) company has ever had by had?

---

## Content -
1. **Problems that require Machine Learning.**
2. **Process of Machine Learning.**
3. **Tools for Machine Learning.**
4. **Mathematics required for Machine Learning.**

## Let's Begin!
### 1. Problems that require Machine Learning
The problems that require Machine Learning over traditional solutions are categorized as follows:
- Classification
- Regression
- Clustering
- Dimensionality Reduction

These problems are solved using different types of learning algorithms that are broadly classified as follows:
- Supervised Learning
- Unsupervised Learning
- Reinforcement Learning

### 2. Process of Machine Learning
The process of machine learning involves the following steps:
- **Data Collection:** What data exists?... Where can we get it?... Is it public?... Are there privacy concerns?... Is it Structured or Unstructured?... All these questions are the main focus of this step because data is the raw material that our Machine Learning algorithm is going to work with, there is no Machine Learning without any data. Therefore the whole process starts with the collection of relevant data.

- **Data Preparation:** Once we have the data that we'll work on, we need to set it up in a format that is easiest and most efficient for a Machine to crunch on. This is a very important step as the quality of our data determines the quality and performance of our Machine Learning model. This is done by processing the data and discarding any outliers that might wrongly influence our model, we then perform EDA (Exploratory Data Analysis), Data Splitting, etc to clean up our dataset.

- **Train a Model:** Finally, we get to train our machine learning model using the dataset we derived from the previous steps. To do this we choose the most appropriate algorithm based on the data or the problem that is at hand. After that, we try to eliminate any overfitting by regularization and finally we tune the hyperparameters and wait. After this, our model is now trained and ready for the first iteration of testing.

- **Analysis/Evaluation:** Now we test our trained model on some evaluation metrics to measure its quality and try to optimize its training cost. We also try running it on some edge cases to see how efficient it is on those samples. If need be we go back and collect more data or improve our model.

- **Serve Model:** We finally put our model in production and let it crunch heaps of data to validate its performance. So we need to observe the model and re-evaluate it so that we can determine if it's good enough for deployment.

- **Retrain Model:** If our model fails to achieve its performance threshold, we retrain the model and check and see if the old predictions are still valid. This loop continues till we get a satisfactory model.

### 3. Tools for Machine Learning
There is an abundance of tools available for every aspect of Machine Learning, some of them are mentioned below:

**Libraries/Code Space:**
- Jupyter
- TenserFlow.js
- PyTorch
- ONNX
- scikit-learn

**Experiment Tracking:**
- Dashboard by Weights & Biases
- TensorBoard
- neptune.ai

**Pre-trained models:**
- Detectron2
- TensorFlow Hub
- Pytorch Hub
- Hugging Face - Transformers

**Data and Model Tracking:**
- Artifacts by Weights & Biases
- Data Version Control

**Cloud Compute Services:**
- Google Cloud
- Amazon Web Services
- Microsoft Azure

**AutoML & Hyperparameter Tuning:**
- Sweeps by Weights & Biases
- Google Cloud AutoML
- Microsoft Azure AutoML

**ML Lifecycle:**
- Kubeflow
- Seldon
- mlflow

**UI Design:**
- Streamlit

### 4. Mathematics required for Machine Learning
Some of the most important topics in mathematics that are required to have a good understanding of machine learning are as follows:
- Statistics
- Linear Algebra
- Multivariable Calculus
- Probability + Distributions
- Optimization
