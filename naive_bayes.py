import pandas as pd
from sklearn.model_selection import train_test_split

class ConfusionMatrix:

    def __init__(self, actual, predicted):
        self.actual = actual
        self.predicted = predicted
        self.labels = sorted(set(actual + predicted))
        self.matrix = self.build_matrix()
    
    def build_matrix(self):
        matrix = {label: {label: 0 for label in self.labels} for label in self.labels}

        for act, pred in zip(self.actual, self.predicted):
            matrix[act][pred] += 1
        
        return matrix
    def print_matrix(self):
        
        
        print("CONFUSION MATRIX:")
        print(" " * 15 + "Predicted as")
        print(" " * 12 + " ".join(f"{label:>5}" for label in self.labels))
        print(" " * 10 + "-" * (6 * len(self.labels)))

        # Printing rows with actual labels
        for label in self.labels:
            row = " ".join(f"{self.matrix[label][l]:>5}" for l in self.labels)
            print(f"Actual {label:>4} | {row}")


class AttributeTable:

    def __init__(self):
        self.attributeDictionary = {}


    def __str__(self):
        return str(self.attributeDictionary)

class ClassTable:
    
    def __init__(self):
        self.classDictionary = {}

   
    def __str__(self):
        return str(self.classDictionary)

class ProbabilityTable:

    def __init__(self):
        self.classTables = {}
        self.classProbabilities = {}

  
    def __str__(self):
        return str(self.classTables)


P_TABLE = ProbabilityTable()
CLASS_NAME = "Outcome"

TEST_INSTANCES = None

def conditional_probability(df, class_lbl, attr, attr_lbl):
    # P(attr_lbl | class_lbl)
    classCount = df[df[CLASS_NAME] == class_lbl].shape[0]
    attrCount = df[(df[CLASS_NAME] == class_lbl) & (df[attr] == attr_lbl)].shape[0]
    
    return attrCount / classCount if classCount > 0 else 0

def build_class_table(df, class_lbl, attrs):
    classTable = ClassTable()
    for attr in attrs:
        attributeTable = AttributeTable()
        attrLabels = df[attr].unique()
        
        for unique_lbl in attrLabels:
            attributeTable.attributeDictionary[unique_lbl] = conditional_probability(df, class_lbl, attr, unique_lbl)
        
     
        classTable.classDictionary[attr] = attributeTable
    
    return classTable

def build_data(data_location):
    df = pd.read_csv(data_location)
    

    df['Glucose'] = pd.cut(df['Glucose'], bins=4, labels=False)
    df['BloodPressure'] = pd.cut(df['BloodPressure'], bins=10, labels=False)
    df['SkinThickness'] = pd.cut(df['SkinThickness'], bins=5, labels=False)
    df['BMI'] = pd.cut(df['BMI'], bins=5, labels=False)
    df['Age'] = pd.cut(df['Age'], bins=5, labels=False)
    df['Insulin'] = pd.cut(df['Insulin'], bins=5, labels=False)
    df['DiabetesPedigreeFunction'] = pd.cut(df['DiabetesPedigreeFunction'], bins=5, labels=False)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=1)

    global TEST_INSTANCES
    #make TEST_INSTANCES a list of all instances in df
    TEST_INSTANCES = test_df.to_dict(orient='records')
    
    classLabels = train_df[CLASS_NAME].unique()
    otherAttrs = train_df.loc[:, train_df.columns != CLASS_NAME].columns.tolist()
    for lbl in classLabels:
        #print("Looking at class label:", lbl)
        P_TABLE.classProbabilities[lbl] = train_df[train_df[CLASS_NAME] == lbl].shape[0] / train_df.shape[0]
        P_TABLE.classTables[lbl] = build_class_table(train_df, lbl, otherAttrs)
    #print(f"Total Training Instances: {train_df.shape[0]}")

def classify_instances(instances):
    correct = 0
    total = 0
    actuals, predicted = [], []
    for instance in instances:
        total += 1
        #print("hi")
        #print(instance)
        bestClass = None
        highestProbability = -1
        for classLabel in P_TABLE.classProbabilities:
           # print(f"\nClass: {classLabel}")
           # print(f"  P({classLabel}) = {P_TABLE.classProbabilities[classLabel]}")
            classTable = P_TABLE.classTables[classLabel].classDictionary
            totalProbability = P_TABLE.classProbabilities[classLabel]
            for attr in classTable:
                #print(f"  Attribute: {attr}")
                #print(classTable[attr].attributeDictionary)
                if instance[attr] not in classTable[attr].attributeDictionary:
                    #print(f"    P({instance[attr]} | {classLabel}) = 0")
                    totalProbability = 0
                    break
                probability = classTable[attr].attributeDictionary[instance[attr]]
              #  print(f"    P({instance[attr]} | {classLabel}) = {probability}")
                totalProbability *= probability
            if totalProbability > highestProbability:
                highestProbability = totalProbability
                bestClass = classLabel
           # print(f"  Total Probability: {totalProbability}")
        #print(f"Chosen classification: {bestClass}")

        if bestClass == instance[CLASS_NAME]:
            correct += 1
        actuals.append(instance[CLASS_NAME])
        predicted.append(bestClass)
    confusion_matrix = ConfusionMatrix(actuals, predicted)
    confusion_matrix.print_matrix()
    print()
    print(f"Total Test Instances: {total}")
    print(f"Correctly Classified: {correct} ({correct / total * 100}%)")
    return confusion_matrix, P_TABLE

def main():
    data_location = "../data/diabetes.csv"
    build_data(data_location)
    
    '''for classTableKey in P_TABLE.classTables:
        print(f"\nClass: {classTableKey}")
        table = P_TABLE.classTables[classTableKey].classDictionary
        for attr in table:
            print(f"  Attribute: {attr}")
            print(f"    {table[attr]}") '''
        

    classify_instances(TEST_INSTANCES)

if __name__ == "__main__":
    main()
